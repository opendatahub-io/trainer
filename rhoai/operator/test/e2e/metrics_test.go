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
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const serviceAccountName = "odh-trainer-operator-controller-manager"
const metricsServiceName = "odh-trainer-operator-controller-manager-metrics-service"
const metricsRoleBindingName = "odh-trainer-operator-metrics-binding"

func TestMetricsEndpoint(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics")

	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: metricsRoleBindingName,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "odh-trainer-operator-metrics-reader",
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      serviceAccountName,
				Namespace: namespace,
			},
		},
	}
	_, err := k8sClient.RbacV1().ClusterRoleBindings().Create(ctx, crb, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create ClusterRoleBinding")

	_, err = k8sClient.CoreV1().Services(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Metrics service should exist")

	token := serviceAccountToken(t)
	g.Expect(token).NotTo(BeEmpty())

	verifyMetricsEndpointReady := func(g Gomega) {
		endpoints, err := k8sClient.CoreV1().Endpoints(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		found := false
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) == 0 {
				continue
			}
			for _, port := range subset.Ports {
				if port.Port == 8443 {
					found = true
				}
			}
		}
		g.Expect(found).To(BeTrue(), "Metrics endpoint is not ready")
	}
	g.Eventually(verifyMetricsEndpointReady).Should(Succeed())

	verifyMetricsServerStarted := func(g Gomega) {
		output, err := k8sClient.GetControllerLogs(ctx, namespace)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(output).To(ContainSubstring("controller-runtime.metrics\tServing metrics server"),
			"Metrics server not yet started")
	}
	g.Eventually(verifyMetricsServerStarted).Should(Succeed())

	runAsNonRoot := true
	var runAsUser int64 = 1000
	allowPrivilegeEscalation := false
	curlPodSpec := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "curl-metrics",
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			RestartPolicy:      corev1.RestartPolicyNever,
			ServiceAccountName: serviceAccountName,
			Containers: []corev1.Container{
				{
					Name:    "curl",
					Image:   "curlimages/curl:latest",
					Command: []string{"/bin/sh", "-c"},
					Args: []string{
						fmt.Sprintf(
							"curl -v -k -H 'Authorization: Bearer %s' https://%s.%s.svc.cluster.local:8443/metrics",
							token, metricsServiceName, namespace,
						),
					},
					SecurityContext: &corev1.SecurityContext{
						AllowPrivilegeEscalation: &allowPrivilegeEscalation,
						Capabilities: &corev1.Capabilities{
							Drop: []corev1.Capability{"ALL"},
						},
						RunAsNonRoot: &runAsNonRoot,
						RunAsUser:    &runAsUser,
						SeccompProfile: &corev1.SeccompProfile{
							Type: corev1.SeccompProfileTypeRuntimeDefault,
						},
					},
				},
			},
		},
	}

	verifyCurlMetrics := func(g Gomega) {
		pod, err := k8sClient.CoreV1().Pods(namespace).Get(ctx, "curl-metrics", metav1.GetOptions{})
		if err != nil {
			_, createErr := k8sClient.CoreV1().Pods(namespace).Create(ctx, curlPodSpec, metav1.CreateOptions{})
			g.Expect(createErr).NotTo(HaveOccurred(), "Failed to create curl-metrics pod")
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		if pod.Status.Phase == corev1.PodFailed {
			_ = k8sClient.CoreV1().Pods(namespace).Delete(ctx, "curl-metrics", metav1.DeleteOptions{})
			g.Expect(pod.Status.Phase).To(Equal(corev1.PodSucceeded), "curl pod failed, retrying")
			return
		}
		g.Expect(pod.Status.Phase).To(Equal(corev1.PodSucceeded), "curl pod in wrong status")
	}
	g.Eventually(verifyCurlMetrics, 5*time.Minute).Should(Succeed())

	metricsOutput := getMetricsOutput(t)
	g.Expect(metricsOutput).To(ContainSubstring(
		"controller_runtime_reconcile_total",
	))
}

func serviceAccountToken(t *testing.T) string {
	t.Helper()
	g := NewWithT(t)

	var out string
	verifyTokenCreation := func(g Gomega) {
		tokenReq, err := k8sClient.CoreV1().ServiceAccounts(namespace).CreateToken(
			ctx,
			serviceAccountName,
			&authenticationv1.TokenRequest{},
			metav1.CreateOptions{},
		)
		g.Expect(err).NotTo(HaveOccurred())
		out = tokenReq.Status.Token
	}
	g.Eventually(verifyTokenCreation).Should(Succeed())

	return out
}

func getMetricsOutput(t *testing.T) string {
	t.Helper()
	g := NewWithT(t)

	metricsOutput, err := k8sClient.GetPodLogs(ctx, "curl-metrics", namespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve logs from curl pod")
	g.Expect(metricsOutput).To(ContainSubstring("< HTTP/1.1 200 OK"))
	return metricsOutput
}
