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

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
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

const (
	trainerNamespace = "opendatahub"
	trainerPartOf    = "trainer"
	platformPartOf   = "platform.opendatahub.io/part-of"

	jobSetCRDName      = "jobsets.jobset.x-k8s.io"
	jobSetVersion      = "v1alpha2"
	jobSetResourceName = "jobsets"

	crdOpenAPI = "object"
)

func TestTrainerModuleLifecycle(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	err := k8sClient.CreateTrainer(ctx, common.Managed, trainerNamespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	t.Cleanup(func() {
		_ = k8sClient.DeleteTrainer(ctx)
	})

	// Phase 1: Managed → Ready
	verifyManagedReady := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.ObservedGeneration).To(Equal(trainer.Generation))
		g.Expect(trainer.Status.Phase).To(Equal(common.PhaseReady))

		readyCond := findCondition(trainer, common.ConditionTypeReady)
		g.Expect(readyCond).NotTo(BeNil())
		g.Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))

		provCond := findCondition(trainer, common.ConditionTypeProvisioningSucceeded)
		g.Expect(provCond).NotTo(BeNil())
		g.Expect(provCond.Status).To(Equal(metav1.ConditionTrue))
		g.Expect(provCond.Reason).To(Equal("Provisioned"))

		g.Expect(trainer.Status.Releases).NotTo(BeEmpty())
		g.Expect(trainer.Status.Releases[0].Name).To(Equal("Kubeflow Trainer"))
	}
	g.Eventually(verifyManagedReady).Should(Succeed())

	ns, err := k8sClient.CoreV1().Namespaces().Get(ctx, trainerNamespace, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Trainer namespace should exist")
	g.Expect(ns.Labels).To(HaveKeyWithValue(platformPartOf, trainerPartOf))

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
	g.Expect(ctrNames).To(HaveLen(15), "Expected 15 ClusterTrainingRuntimes")

	// Phase 2: Managed → Removed
	err = k8sClient.UpdateTrainerManagementState(ctx, common.Removed)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to update Trainer to Removed")

	verifyRemoved := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.Phase).To(Equal(common.PhaseNotReady))

		provCond := findCondition(trainer, common.ConditionTypeProvisioningSucceeded)
		g.Expect(provCond).NotTo(BeNil())
		g.Expect(provCond.Status).To(Equal(metav1.ConditionFalse))
		g.Expect(provCond.Reason).To(Equal("Removed"))
	}
	g.Eventually(verifyRemoved).Should(Succeed())

	verifyResourcesCleanedUp := func(g Gomega) {
		deps, err := k8sClient.ListDeployments(ctx, trainerNamespace, platformPartOf+"="+trainerPartOf)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(deps).To(BeEmpty(), "Deployments should be cleaned up after Removed")

		ctrs, err := k8sClient.ListClusterTrainingRuntimes(ctx, platformPartOf+"="+trainerPartOf)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(ctrs).To(BeEmpty(), "ClusterTrainingRuntimes should be cleaned up after Removed")
	}
	g.Eventually(verifyResourcesCleanedUp).Should(Succeed())

	// Phase 3: Removed → Managed (re-provisioning)
	err = k8sClient.UpdateTrainerManagementState(ctx, common.Managed)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to update Trainer back to Managed")

	g.Eventually(verifyManagedReady).Should(Succeed())

	deployments, err = k8sClient.AppsV1().Deployments(trainerNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: platformPartOf + "=" + trainerPartOf,
	})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(deployments.Items).NotTo(BeEmpty(), "Deployments should be re-created after Managed")
	for _, d := range deployments.Items {
		g.Expect(d.Status.ReadyReplicas).To(Equal(d.Status.Replicas),
			"Deployment %s should have all replicas ready after re-provisioning", d.Name)
	}

	ctrNames, err = k8sClient.ListClusterTrainingRuntimes(ctx, platformPartOf+"="+trainerPartOf)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(ctrNames).NotTo(BeEmpty(), "ClusterTrainingRuntimes should be re-created after Managed")

	// Phase 4: Delete
	err = k8sClient.DeleteTrainer(ctx)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to delete Trainer CR")

	verifyTrainerDeleted := func(g Gomega) {
		_, err := k8sClient.GetTrainer(ctx)
		g.Expect(errors.IsNotFound(err)).To(BeTrue(), "Trainer CR should be deleted")
	}
	g.Eventually(verifyTrainerDeleted).Should(Succeed())
}

// TODO: Convert to OpenShift e2e test. On Kind the upstream trainer controller
// can't start without webhook certs (no service serving cert annotation), so
// the validating webhook blocks CTR updates. On OpenShift, deploy Trainer CR,
// create a TrainJob referencing torch-distributed-cpu to trigger the
// resource-in-use finalizer, then delete the Trainer CR and verify it completes
// without being blocked by the stuck CTR.

func findCondition(trainer *componentsv1alpha1.Trainer, condType common.ConditionType) *common.Condition {
	for i := range trainer.Status.Conditions {
		if trainer.Status.Conditions[i].Type == string(condType) {
			return &trainer.Status.Conditions[i]
		}
	}
	return nil
}

// +kubebuilder:scaffold:e2e-webhooks-checks
