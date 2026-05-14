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
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	"github.com/hrathina/odh-trainer-operator/test/utils"
)

// namespace where the project is deployed in
const namespace = "odh-trainer-operator-system"

// serviceAccountName created for the project
const serviceAccountName = "odh-trainer-operator-controller-manager"

// metricsServiceName is the name of the metrics service of the project
const metricsServiceName = "odh-trainer-operator-controller-manager-metrics-service"

// metricsRoleBindingName is the name of the RBAC that will be created to allow get the metrics data
const metricsRoleBindingName = "odh-trainer-operator-metrics-binding"

func TestManager(t *testing.T) {
	g := NewWithT(t)

	var controllerPodName string

	// Setup: create namespace, label it, install CRDs, deploy controller
	cmd := exec.Command("kubectl", "create", "ns", namespace)
	_, err := utils.Run(cmd)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create namespace")

	cmd = exec.Command("kubectl", "label", "--overwrite", "ns", namespace,
		"pod-security.kubernetes.io/enforce=restricted")
	_, err = utils.Run(cmd)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to label namespace with restricted policy")

	cmd = exec.Command("make", "install")
	_, err = utils.Run(cmd)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to install CRDs")

	cmd = exec.Command("make", "deploy", fmt.Sprintf("IMG=%s", projectImage))
	_, err = utils.Run(cmd)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to deploy the controller-manager")

	t.Cleanup(func() {
		cmd := exec.Command("kubectl", "delete", "pod", "curl-metrics", "-n", namespace)
		_, _ = utils.Run(cmd)

		cmd = exec.Command("make", "undeploy")
		_, _ = utils.Run(cmd)

		cmd = exec.Command("make", "uninstall")
		_, _ = utils.Run(cmd)

		cmd = exec.Command("kubectl", "delete", "ns", namespace)
		_, _ = utils.Run(cmd)
	})

	t.Run("should run successfully", func(t *testing.T) {
		g := NewWithT(t)
		registerDebugCleanup(t, &controllerPodName)

		verifyControllerUp := func(g Gomega) {
			cmd := exec.Command("kubectl", "get",
				"pods", "-l", "control-plane=controller-manager",
				"-o", "go-template={{ range .items }}"+
					"{{ if not .metadata.deletionTimestamp }}"+
					"{{ .metadata.name }}"+
					"{{ \"\\n\" }}{{ end }}{{ end }}",
				"-n", namespace,
			)

			podOutput, err := utils.Run(cmd)
			g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve controller-manager pod information")
			podNames := utils.GetNonEmptyLines(podOutput)
			g.Expect(podNames).To(HaveLen(1), "expected 1 controller pod running")
			controllerPodName = podNames[0]
			g.Expect(controllerPodName).To(ContainSubstring("controller-manager"))

			cmd = exec.Command("kubectl", "get",
				"pods", controllerPodName, "-o", "jsonpath={.status.phase}",
				"-n", namespace,
			)
			output, err := utils.Run(cmd)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(output).To(Equal("Running"), "Incorrect controller-manager pod status")
		}
		g.Eventually(verifyControllerUp).Should(Succeed())
	})

	t.Run("should ensure the metrics endpoint is serving metrics", func(t *testing.T) {
		g := NewWithT(t)
		registerDebugCleanup(t, &controllerPodName)

		cmd := exec.Command("kubectl", "create", "clusterrolebinding", metricsRoleBindingName,
			"--clusterrole=odh-trainer-operator-metrics-reader",
			fmt.Sprintf("--serviceaccount=%s:%s", namespace, serviceAccountName),
		)
		_, err := utils.Run(cmd)
		g.Expect(err).NotTo(HaveOccurred(), "Failed to create ClusterRoleBinding")

		cmd = exec.Command("kubectl", "get", "service", metricsServiceName, "-n", namespace)
		_, err = utils.Run(cmd)
		g.Expect(err).NotTo(HaveOccurred(), "Metrics service should exist")

		token, err := serviceAccountToken(t)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(token).NotTo(BeEmpty())

		verifyMetricsEndpointReady := func(g Gomega) {
			cmd := exec.Command("kubectl", "get", "endpoints", metricsServiceName, "-n", namespace)
			output, err := utils.Run(cmd)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(output).To(ContainSubstring("8443"), "Metrics endpoint is not ready")
		}
		g.Eventually(verifyMetricsEndpointReady).Should(Succeed())

		verifyMetricsServerStarted := func(g Gomega) {
			cmd := exec.Command("kubectl", "logs", controllerPodName, "-n", namespace)
			output, err := utils.Run(cmd)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(output).To(ContainSubstring("controller-runtime.metrics\tServing metrics server"),
				"Metrics server not yet started")
		}
		g.Eventually(verifyMetricsServerStarted).Should(Succeed())

		cmd = exec.Command("kubectl", "run", "curl-metrics", "--restart=Never",
			"--namespace", namespace,
			"--image=curlimages/curl:latest",
			"--overrides",
			fmt.Sprintf(`{
						"spec": {
							"containers": [{
								"name": "curl",
								"image": "curlimages/curl:latest",
								"command": ["/bin/sh", "-c"],
								"args": ["curl -v -k -H 'Authorization: Bearer %s' https://%s.%s.svc.cluster.local:8443/metrics"],
								"securityContext": {
									"allowPrivilegeEscalation": false,
									"capabilities": {
										"drop": ["ALL"]
									},
									"runAsNonRoot": true,
									"runAsUser": 1000,
									"seccompProfile": {
										"type": "RuntimeDefault"
									}
								}
							}],
							"serviceAccount": "%s"
						}
					}`, token, metricsServiceName, namespace, serviceAccountName))
		_, err = utils.Run(cmd)
		g.Expect(err).NotTo(HaveOccurred(), "Failed to create curl-metrics pod")

		verifyCurlUp := func(g Gomega) {
			cmd := exec.Command("kubectl", "get", "pods", "curl-metrics",
				"-o", "jsonpath={.status.phase}",
				"-n", namespace)
			output, err := utils.Run(cmd)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(output).To(Equal("Succeeded"), "curl pod in wrong status")
		}
		g.Eventually(verifyCurlUp, 5*time.Minute).Should(Succeed())

		metricsOutput := getMetricsOutput(t)
		g.Expect(metricsOutput).To(ContainSubstring(
			"controller_runtime_reconcile_total",
		))
	})

	// +kubebuilder:scaffold:e2e-webhooks-checks
}

func registerDebugCleanup(t *testing.T, controllerPodName *string) {
	t.Helper()
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}

		cmd := exec.Command("kubectl", "logs", *controllerPodName, "-n", namespace)
		controllerLogs, err := utils.Run(cmd)
		if err == nil {
			t.Logf("Controller logs:\n %s", controllerLogs)
		} else {
			t.Logf("Failed to get Controller logs: %s", err)
		}

		cmd = exec.Command("kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp")
		eventsOutput, err := utils.Run(cmd)
		if err == nil {
			t.Logf("Kubernetes events:\n%s", eventsOutput)
		} else {
			t.Logf("Failed to get Kubernetes events: %s", err)
		}

		cmd = exec.Command("kubectl", "logs", "curl-metrics", "-n", namespace)
		metricsOutput, err := utils.Run(cmd)
		if err == nil {
			t.Logf("Metrics logs:\n %s", metricsOutput)
		} else {
			t.Logf("Failed to get curl-metrics logs: %s", err)
		}

		cmd = exec.Command("kubectl", "describe", "pod", *controllerPodName, "-n", namespace)
		podDescription, err := utils.Run(cmd)
		if err == nil {
			t.Logf("Pod description:\n%s", podDescription)
		} else {
			t.Log("Failed to describe controller pod")
		}
	})
}

func serviceAccountToken(t *testing.T) (string, error) {
	t.Helper()
	g := NewWithT(t)

	const tokenRequestRawString = `{
		"apiVersion": "authentication.k8s.io/v1",
		"kind": "TokenRequest"
	}`

	secretName := fmt.Sprintf("%s-token-request", serviceAccountName)
	tokenRequestFile := filepath.Join("/tmp", secretName)
	err := os.WriteFile(tokenRequestFile, []byte(tokenRequestRawString), os.FileMode(0o644))
	if err != nil {
		return "", err
	}

	var out string
	verifyTokenCreation := func(g Gomega) {
		cmd := exec.Command("kubectl", "create", "--raw", fmt.Sprintf(
			"/api/v1/namespaces/%s/serviceaccounts/%s/token",
			namespace,
			serviceAccountName,
		), "-f", tokenRequestFile)

		output, err := cmd.CombinedOutput()
		g.Expect(err).NotTo(HaveOccurred())

		var token tokenRequest
		err = json.Unmarshal(output, &token)
		g.Expect(err).NotTo(HaveOccurred())

		out = token.Status.Token
	}
	g.Eventually(verifyTokenCreation).Should(Succeed())

	return out, err
}

func getMetricsOutput(t *testing.T) string {
	t.Helper()
	g := NewWithT(t)

	cmd := exec.Command("kubectl", "logs", "curl-metrics", "-n", namespace)
	metricsOutput, err := utils.Run(cmd)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve logs from curl pod")
	g.Expect(metricsOutput).To(ContainSubstring("< HTTP/1.1 200 OK"))
	return metricsOutput
}

type tokenRequest struct {
	Status struct {
		Token string `json:"token"`
	} `json:"status"`
}
