/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2e

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/trainer-operator/test/support"
	"github.com/opendatahub-io/trainer-operator/test/utils"
)

var (
	skipCertManagerInstall        = os.Getenv("CERT_MANAGER_INSTALL_SKIP") == "true"
	isCertManagerAlreadyInstalled = false

	projectImage = "example.com/odh-trainer-operator:v0.0.1"

	k8sClient *support.Client
	ctx       = context.Background()
)

const (
	namespace             = "trainer-operator-system"
	metricsServiceName    = "trainer-operator-controller-manager-metrics-service"
	metricsReaderRoleName = "trainer-operator-metrics-reader"
	metricsBindingName    = "trainer-operator-metrics-binding"
	metricsTLSSecretName  = "trainer-operator-metrics-tls"
	metricsPort           = 8443
)

func TestMain(m *testing.M) {
	fmt.Fprintln(os.Stderr, "Starting odh-trainer-operator integration test suite")

	gomega.SetDefaultEventuallyTimeout(5 * time.Minute)
	gomega.SetDefaultEventuallyPollingInterval(time.Second)

	cmd := exec.Command("make", "docker-build", fmt.Sprintf("IMG=%s", projectImage))
	if _, err := utils.Run(cmd); err != nil {
		log.Fatalf("Failed to build the manager(Operator) image: %v", err)
	}

	if err := utils.LoadImageToKindClusterWithName(projectImage); err != nil {
		log.Fatalf("Failed to load the manager(Operator) image into Kind: %v", err)
	}

	if !skipCertManagerInstall {
		isCertManagerAlreadyInstalled = utils.IsCertManagerCRDsInstalled()
		if !isCertManagerAlreadyInstalled {
			fmt.Fprintln(os.Stderr, "Installing CertManager...")
			if err := utils.InstallCertManager(); err != nil {
				log.Fatalf("Failed to install CertManager: %v", err)
			}
		} else {
			fmt.Fprintln(os.Stderr, "WARNING: CertManager is already installed. Skipping installation...")
		}
	}

	if !utils.IsPrometheusCRDsInstalled() {
		fmt.Fprintln(os.Stderr, "Installing Prometheus Operator...")
		if err := utils.InstallPrometheusOperator(); err != nil {
			log.Fatalf("Failed to install Prometheus Operator: %v", err)
		}
	}

	var err error
	k8sClient, err = support.NewClient()
	if err != nil {
		log.Fatalf("Failed to create Kubernetes client: %v", err)
	}

	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
			Labels: map[string]string{
				"pod-security.kubernetes.io/enforce": "restricted",
			},
		},
	}
	if _, err := k8sClient.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil {
		log.Fatalf("Failed to create namespace: %v", err)
	}

	if err := createMetricsTLSSecret(); err != nil {
		log.Fatalf("Failed to create metrics TLS secret: %v", err)
	}

	cmd = exec.Command("make", "install")
	if _, err := utils.Run(cmd); err != nil {
		log.Fatalf("Failed to install CRDs: %v", err)
	}

	if err := utils.InstallImageStreamCRD(); err != nil {
		log.Fatalf("Failed to install ImageStream CRD: %v", err)
	}

	cmd = exec.Command("make", "deploy", fmt.Sprintf("IMG=%s", projectImage))
	if _, err := utils.Run(cmd); err != nil {
		log.Fatalf("Failed to deploy the controller-manager: %v", err)
	}

	if err := installJobSetCRD(); err != nil {
		log.Fatalf("Failed to install JobSet CRD: %v", err)
	}

	code := m.Run()

	_ = k8sClient.DeleteTrainer(ctx)
	_ = k8sClient.ApiextensionsClient.ApiextensionsV1().CustomResourceDefinitions().Delete(
		ctx, jobSetCRDName, metav1.DeleteOptions{})
	_ = k8sClient.CoreV1().Namespaces().Delete(ctx, "opendatahub", metav1.DeleteOptions{})
	_ = k8sClient.CoreV1().Pods(namespace).Delete(ctx, "curl-metrics", metav1.DeleteOptions{})
	_ = k8sClient.RbacV1().ClusterRoleBindings().Delete(
		ctx, "trainer-operator-metrics-binding", metav1.DeleteOptions{})

	cmd = exec.Command("make", "undeploy")
	_, _ = utils.Run(cmd)

	cmd = exec.Command("make", "uninstall")
	_, _ = utils.Run(cmd)

	_ = k8sClient.CoreV1().Namespaces().Delete(ctx, namespace, metav1.DeleteOptions{})

	utils.UninstallPrometheusOperator()

	if !skipCertManagerInstall && !isCertManagerAlreadyInstalled {
		fmt.Fprintln(os.Stderr, "Uninstalling CertManager...")
		utils.UninstallCertManager()
	}

	os.Exit(code)
}

func installJobSetCRD() error {
	crdClient := k8sClient.ApiextensionsClient.ApiextensionsV1().CustomResourceDefinitions()

	jobSetCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "jobset.x-k8s.io",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   "JobSet",
				Plural: jobSetResourceName,
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    jobSetVersion,
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: crdOpenAPI,
						},
					},
				},
			},
		},
	}

	if _, err := crdClient.Create(ctx, jobSetCRD, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create JobSet CRD: %w", err)
	}

	return nil
}

func createMetricsTLSSecret() error {
	crt, key, err := generateSelfSignedCert()
	if err != nil {
		return fmt.Errorf("generating metrics serving cert: %w", err)
	}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      metricsTLSSecretName,
			Namespace: namespace,
		},
		Type: corev1.SecretTypeTLS,
		Data: map[string][]byte{
			"tls.crt": crt,
			"tls.key": key,
		},
	}
	if _, err := k8sClient.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("creating metrics TLS secret: %w", err)
	}
	return nil
}

func generateSelfSignedCert() (crt, key []byte, err error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("generating key: %w", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		DNSNames:              metricsServiceDNSNames(),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("creating certificate: %w", err)
	}
	keyBytes, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		return nil, nil, fmt.Errorf("marshalling key: %w", err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}),
		nil
}

func metricsServiceDNSNames() []string {
	return []string{
		metricsServiceName + "." + namespace + ".svc",
		metricsServiceName + "." + namespace + ".svc.cluster.local",
	}
}
