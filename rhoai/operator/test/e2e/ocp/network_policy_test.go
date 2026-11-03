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

package ocp

import (
	"net"
	"strconv"
	"testing"

	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const curlContainerName = "curl"
const curlContainerImage = "curlimages/curl:8.12.1"

func TestNetworkPolicyBlocksMetricsFromOtherNamespace(t *testing.T) {
	controllerPod := getReadyControllerPod(t)
	metricsPort := lookupPortByName(t, controllerPod, "metrics")
	assertCrossNamespaceConnectionTimesOut(t, controllerPod, metricsPort)
}

func TestNetworkPolicyBlocksUnpublishedPortFromOtherNamespace(t *testing.T) {
	assertCrossNamespaceConnectionTimesOut(t, getReadyControllerPod(t), 31415)
}

func assertCrossNamespaceConnectionTimesOut(t *testing.T, controllerPod corev1.Pod, port int32) {
	t.Helper()
	probeNamespace := k8sClient.CreateTestNamespace(t, "trainer-operator-netpol-").Name
	url := "https://" + net.JoinHostPort(controllerPod.Status.PodIP, strconv.Itoa(int(port)))
	logs, exitCode := runCurlProbe(t, probeNamespace, url)
	g := gomega.NewWithT(t)

	g.Expect(exitCode).To(gomega.Equal(int32(28)), "curl logs: %s", logs)
	g.Expect(logs).To(gomega.ContainSubstring("CURL_TIMING time_connect=0.000000 "),
		"expected curl to time out before establishing a TCP connection; logs: %s", logs)
}

func runCurlProbe(t *testing.T, probeNamespace, url string) (string, int32) {
	t.Helper()
	pod, err := k8sClient.CoreV1().Pods(probeNamespace).Create(t.Context(), &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "curl-probe-"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name: curlContainerName, Image: curlContainerImage, Command: []string{"curl"},
				Args: []string{
					"--silent", "--show-error", "--verbose", "--noproxy", "*",
					"--connect-timeout", "5", "--max-time", "10", "--insecure",
					"--output", "/dev/null", "--write-out",
					"\nCURL_TIMING time_connect=%{time_connect} time_total=%{time_total}\n", url,
				},
			}},
		},
	}, metav1.CreateOptions{})
	g := gomega.NewWithT(t)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	var terminated *corev1.ContainerStateTerminated
	g.Eventually(func(g gomega.Gomega) {
		current, err := k8sClient.CoreV1().Pods(probeNamespace).Get(t.Context(), pod.Name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		pod = current
		terminated = nil
		for _, status := range current.Status.ContainerStatuses {
			if status.Name == curlContainerName {
				terminated = status.State.Terminated
			}
		}
		g.Expect(terminated).NotTo(gomega.BeNil(), "curl pod status: %+v", current.Status)
	}).Should(gomega.Succeed())

	logs, err := k8sClient.GetPodLogs(t.Context(), pod.Name, probeNamespace)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	t.Logf("Curl pod %s/%s exit=%d target=%s logs:\n%s",
		probeNamespace, pod.Name, terminated.ExitCode, url, logs)
	return logs, terminated.ExitCode
}

func lookupPortByName(t *testing.T, pod corev1.Pod, portName string) int32 {
	t.Helper()
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.Name == portName {
				return port.ContainerPort
			}
		}
	}
	t.Fatalf("pod %s has no port named %q", pod.Name, portName)
	return 0
}

func getReadyControllerPod(t *testing.T) corev1.Pod {
	t.Helper()
	pods, err := k8sClient.GetControllerPods(t.Context(), namespace)
	gomega.NewWithT(t).Expect(err).NotTo(gomega.HaveOccurred())
	for _, pod := range pods {
		if pod.Status.PodIP == "" || pod.DeletionTimestamp != nil {
			continue
		}
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				return pod
			}
		}
	}
	t.Fatal("no Ready controller pod with an IP found")
	return corev1.Pod{}
}
