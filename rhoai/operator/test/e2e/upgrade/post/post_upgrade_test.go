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

package post

import (
	"testing"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

const (
	trainerNamespace = "opendatahub"
	platformPartOf   = "platform.opendatahub.io/part-of"
	trainerPartOf    = "trainer"
)

func TestPostUpgradeControllerRunning(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugDumpOnFailure(t, ctx, namespace)

	g.Eventually(func(g Gomega) {
		pods, err := k8sClient.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "control-plane=controller-manager",
		})
		g.Expect(err).NotTo(HaveOccurred())

		var activePods []corev1.Pod
		for _, p := range pods.Items {
			if p.DeletionTimestamp == nil {
				activePods = append(activePods, p)
			}
		}
		g.Expect(activePods).To(HaveLen(1), "expected 1 controller pod running")
		g.Expect(activePods[0].Name).To(ContainSubstring("controller-manager"))
		g.Expect(string(activePods[0].Status.Phase)).To(Equal("Running"))
	}).Should(Succeed())
}

func TestPostUpgradeTrainerStillReady(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugDumpOnFailure(t, ctx, namespace)

	g.Eventually(func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.Phase).To(Equal(common.PhaseReady))
		g.Expect(trainer.Status.ObservedGeneration).To(Equal(trainer.Generation))

		var readyFound, provFound bool
		for i := range trainer.Status.Conditions {
			c := &trainer.Status.Conditions[i]
			switch common.ConditionType(c.Type) {
			case common.ConditionTypeReady:
				g.Expect(c.Status).To(Equal(metav1.ConditionTrue))
				readyFound = true
			case common.ConditionTypeProvisioningSucceeded:
				g.Expect(c.Status).To(Equal(metav1.ConditionTrue))
				provFound = true
			}
		}
		g.Expect(readyFound).To(BeTrue(), "Ready condition not found")
		g.Expect(provFound).To(BeTrue(), "ProvisioningSucceeded condition not found")
	}).Should(Succeed())

	deployments, err := k8sClient.AppsV1().Deployments(trainerNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: platformPartOf + "=" + trainerPartOf,
	})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(deployments.Items).NotTo(BeEmpty(), "Expected at least one Trainer deployment")
	for _, d := range deployments.Items {
		g.Expect(d.Status.ReadyReplicas).To(Equal(d.Status.Replicas),
			"Deployment %s should have all replicas ready", d.Name)
	}

	ctrNames, err := k8sClient.ListClusterTrainingRuntimes(ctx, platformPartOf+"="+trainerPartOf)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(ctrNames).NotTo(BeEmpty(), "Expected ClusterTrainingRuntimes to still exist after upgrade")
}
