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
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMetricsEndpoint(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics")

	waitForMetricsEndpointReady(g)
	ensureMetricsReaderBinding(g)

	output := runCurlPod(g, "curl-metrics", authorizedMetricsCurlScript(), "")
	g.Expect(output).Should(ContainSubstring("HTTP_STATUS:200"))
	g.Expect(output).Should(ContainSubstring("controller_runtime_reconcile_total"))
}

func TestMetricsEndpoint_UnauthenticatedRequest(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics-unauth")

	waitForMetricsEndpointReady(g)

	output := runCurlPod(g, "curl-metrics-unauth", unauthenticatedMetricsCurlScript(), "")
	g.Expect(output).Should(ContainSubstring("HTTP_STATUS:401"))
}

func TestMetricsEndpoint_UnauthorizedServiceAccount(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics-forbidden")

	waitForMetricsEndpointReady(g)
	ensureUnauthorizedServiceAccount(g)

	output := runCurlPod(g, "curl-metrics-forbidden", unauthorizedMetricsCurlScript(), "metrics-unauthorized")
	g.Expect(output).Should(ContainSubstring("HTTP_STATUS:403"))
}

func TestMetricsEndpoint_RejectsTLS11(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace, "curl-metrics-tls11")

	waitForMetricsEndpointReady(g)
	ensureMetricsReaderBinding(g)

	output := runCurlPod(g, "curl-metrics-tls11", tls11RejectedCurlScript(), "")
	g.Expect(output).Should(ContainSubstring("TLS_1_1_REJECTED"))
}

func waitForMetricsEndpointReady(g Gomega) {
	_, err := k8sClient.CoreV1().Services(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
	g.Expect(err).ShouldNot(HaveOccurred(), "Metrics service should exist")

	verifyMetricsEndpointReady := func(g Gomega) {
		endpoints, err := k8sClient.CoreV1().Endpoints(namespace).Get(ctx, metricsServiceName, metav1.GetOptions{})
		g.Expect(err).ShouldNot(HaveOccurred())
		found := false
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) == 0 {
				continue
			}
			for _, port := range subset.Ports {
				if port.Port == metricsPort {
					found = true
				}
			}
		}
		g.Expect(found).Should(BeTrue(), "Metrics endpoint is not ready")
	}
	g.Eventually(verifyMetricsEndpointReady).Should(Succeed())

	verifyMetricsServerStarted := func(g Gomega) {
		output, err := k8sClient.GetControllerLogs(ctx, namespace)
		g.Expect(err).ShouldNot(HaveOccurred())
		g.Expect(output).Should(ContainSubstring("controller-runtime.metrics\tServing metrics server"),
			"Metrics server not yet started")
	}
	g.Eventually(verifyMetricsServerStarted).Should(Succeed())
}

func metricsServiceFQDN() string {
	return metricsServiceName + "." + namespace + ".svc.cluster.local"
}

func authorizedMetricsCurlScript() string {
	return fmt.Sprintf(`TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
for i in $(seq 1 10); do
  code=$(curl -sS --fail --connect-timeout 5 --max-time 10 \
    --cacert /etc/metrics-ca/ca.crt -H "Authorization: Bearer ${TOKEN}" \
    -o /tmp/metrics.out -w "%%{http_code}" \
    https://%s:%d/metrics)
  if [ "${code}" = "200" ]; then
    cat /tmp/metrics.out
    echo "HTTP_STATUS:${code}"
    exit 0
  fi
  sleep 5
done
exit 1`, metricsServiceFQDN(), metricsPort)
}

func unauthenticatedMetricsCurlScript() string {
	return fmt.Sprintf(`for i in $(seq 1 10); do
  code=$(curl -sS --connect-timeout 5 --max-time 10 \
    --cacert /etc/metrics-ca/ca.crt \
    -o /dev/null -w "%%{http_code}" \
    https://%s:%d/metrics)
  if [ "${code}" = "401" ]; then
    echo "HTTP_STATUS:${code}"
    exit 0
  fi
  sleep 5
done
exit 1`, metricsServiceFQDN(), metricsPort)
}

func unauthorizedMetricsCurlScript() string {
	return fmt.Sprintf(`TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
for i in $(seq 1 10); do
  code=$(curl -sS --connect-timeout 5 --max-time 10 \
    --cacert /etc/metrics-ca/ca.crt -H "Authorization: Bearer ${TOKEN}" \
    -o /dev/null -w "%%{http_code}" \
    https://%s:%d/metrics)
  if [ "${code}" = "403" ]; then
    echo "HTTP_STATUS:${code}"
    exit 0
  fi
  sleep 5
done
exit 1`, metricsServiceFQDN(), metricsPort)
}

func tls11RejectedCurlScript() string {
	return fmt.Sprintf(`TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
if curl -sS --connect-timeout 5 --max-time 10 --tlsv1.1 --tls-max 1.1 \
  --cacert /etc/metrics-ca/ca.crt -H "Authorization: Bearer ${TOKEN}" \
  -o /dev/null https://%s:%d/metrics 2>/dev/null; then
  echo "TLS_1_1_UNEXPECTEDLY_SUCCEEDED"
  exit 1
fi
echo "TLS_1_1_REJECTED"
exit 0`, metricsServiceFQDN(), metricsPort)
}

func runCurlPod(g Gomega, podName, script, serviceAccountName string) string {
	podSpec := curlPodSpec(podName, script, serviceAccountName)
	verifyCurlPod := func(g Gomega) {
		pod, err := k8sClient.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			_, createErr := k8sClient.CoreV1().Pods(namespace).Create(ctx, podSpec, metav1.CreateOptions{})
			g.Expect(createErr).ShouldNot(HaveOccurred(), "Failed to create %s pod", podName)
			g.Expect(err).ShouldNot(HaveOccurred())
			return
		}
		if pod.Status.Phase == corev1.PodFailed {
			_ = k8sClient.CoreV1().Pods(namespace).Delete(ctx, podName, metav1.DeleteOptions{})
			g.Expect(pod.Status.Phase).Should(Equal(corev1.PodSucceeded), "%s pod failed, retrying", podName)
			return
		}
		g.Expect(pod.Status.Phase).Should(Equal(corev1.PodSucceeded), "%s pod in wrong status", podName)
	}
	g.Eventually(verifyCurlPod, 5*time.Minute).Should(Succeed())

	output, err := k8sClient.GetPodLogs(ctx, podName, namespace)
	g.Expect(err).ShouldNot(HaveOccurred(), "Failed to retrieve logs from %s pod", podName)
	return output
}

func curlPodSpec(podName, script, serviceAccountName string) *corev1.Pod {
	runAsNonRoot := true
	var runAsUser int64 = 1000
	allowPrivilegeEscalation := false
	spec := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:    "curl",
					Image:   "curlimages/curl:latest",
					Command: []string{"/bin/sh", "-c"},
					Args:    []string{script},
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
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "metrics-ca",
							MountPath: "/etc/metrics-ca",
							ReadOnly:  true,
						},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "metrics-ca",
					VolumeSource: corev1.VolumeSource{
						Secret: &corev1.SecretVolumeSource{
							SecretName: metricsTLSSecretName,
							Items: []corev1.KeyToPath{
								{Key: "tls.crt", Path: "ca.crt"},
							},
						},
					},
				},
			},
		},
	}
	if serviceAccountName != "" {
		spec.Spec.ServiceAccountName = serviceAccountName
	}
	return spec
}

func ensureMetricsReaderBinding(g Gomega) {
	binding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: metricsBindingName,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     metricsReaderRoleName,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      rbacv1.ServiceAccountKind,
				Name:      "default",
				Namespace: namespace,
			},
		},
	}
	_, err := k8sClient.RbacV1().ClusterRoleBindings().Create(ctx, binding, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		g.Expect(err).ShouldNot(HaveOccurred(), "Failed to create metrics reader binding")
	}
}

func ensureUnauthorizedServiceAccount(g Gomega) {
	const saName = "metrics-unauthorized"
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: namespace,
		},
	}
	_, err := k8sClient.CoreV1().ServiceAccounts(namespace).Create(ctx, sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		g.Expect(err).ShouldNot(HaveOccurred(), "Failed to create unauthorized service account")
	}
}
