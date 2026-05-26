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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const metricsServiceName = "odh-trainer-operator-controller-manager-metrics-service"

func TestMetricsEndpoint(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics")

	_, err := k8sClient.CoreV1().Services(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Metrics service should exist")

	verifyMetricsEndpointReady := func(g Gomega) {
		endpoints, err := k8sClient.CoreV1().Endpoints(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		found := false
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) == 0 {
				continue
			}
			for _, port := range subset.Ports {
				if port.Port == 8080 {
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
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:    "curl",
					Image:   "curlimages/curl:latest",
					Command: []string{"/bin/sh", "-c"},
					Args: []string{
						fmt.Sprintf(
							"curl -v http://%s.%s.svc.cluster.local:8080/metrics",
							metricsServiceName, namespace,
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

func getMetricsOutput(t *testing.T) string {
	t.Helper()
	g := NewWithT(t)

	metricsOutput, err := k8sClient.GetPodLogs(ctx, "curl-metrics", namespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve logs from curl pod")
	g.Expect(metricsOutput).To(ContainSubstring("< HTTP/1.1 200 OK"))
	return metricsOutput
}
