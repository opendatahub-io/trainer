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
	"testing"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

func TestControllerPodRunning(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	var podName string
	verifyControllerUp := func(g Gomega) {
		pods, err := k8sClient.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "control-plane=controller-manager",
		})
		g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve controller-manager pod information")

		var activePods []corev1.Pod
		for _, p := range pods.Items {
			if p.DeletionTimestamp == nil {
				activePods = append(activePods, p)
			}
		}
		g.Expect(activePods).To(HaveLen(1), "expected 1 controller pod running")
		podName = activePods[0].Name
		g.Expect(podName).To(ContainSubstring("controller-manager"))
		g.Expect(string(activePods[0].Status.Phase)).To(Equal("Running"), "Incorrect controller-manager pod status")
	}
	g.Eventually(verifyControllerUp).Should(Succeed())
}

const trainerNamespace = "opendatahub"

func TestTrainerReconciliation(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	err := k8sClient.CreateTrainer(ctx, common.Managed, trainerNamespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	t.Cleanup(func() {
		_ = k8sClient.DeleteTrainer(ctx)
	})

	verifyTrainerReconciled := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.ObservedGeneration).To(Equal(trainer.Generation))
	}
	g.Eventually(verifyTrainerReconciled).Should(Succeed())

	ns, err := k8sClient.CoreV1().Namespaces().Get(ctx, trainerNamespace, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Trainer namespace should exist")
	g.Expect(ns.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", "trainer"))

	deployments, err := k8sClient.AppsV1().Deployments(trainerNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: "platform.opendatahub.io/part-of=trainer",
	})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(deployments.Items).NotTo(BeEmpty(), "Expected at least one Trainer deployment")

	err = k8sClient.DeleteTrainer(ctx)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to delete Trainer CR")

	verifyTrainerDeleted := func(g Gomega) {
		_, err := k8sClient.GetTrainer(ctx)
		g.Expect(errors.IsNotFound(err)).To(BeTrue(), "Trainer CR should be deleted")
	}
	g.Eventually(verifyTrainerDeleted).Should(Succeed())
}

// +kubebuilder:scaffold:e2e-webhooks-checks
