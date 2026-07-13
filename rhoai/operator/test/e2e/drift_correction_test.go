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
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

const (
	driftCorrectionTimeout = 2 * time.Minute
)

// TestDriftCorrection verifies that the controller automatically recreates
// managed resources when they are deleted (drift correction via watches).
//
// Test flow:
// 1. Create Trainer CR (Managed state)
// 2. Wait for resources to be created
// 3. Delete a managed ConfigMap
// 4. Verify controller automatically recreates it within timeout
//
// Note: We test with ConfigMap instead of Deployment because deleting the
// Deployment would break the webhook service, preventing reconciliation.
func TestDriftCorrection(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	// Create Trainer CR
	err := k8sClient.CreateTrainer(ctx, trainerNamespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	t.Cleanup(func() {
		_ = k8sClient.DeleteTrainer(ctx)
		g.Eventually(func(g Gomega) {
			_, err := k8sClient.GetTrainer(ctx)
			g.Expect(errors.IsNotFound(err)).To(BeTrue())
		}).WithTimeout(30 * time.Second).Should(Succeed())
	})

	// Wait for Trainer to reach Ready state and resources to be created
	const expectedConfigMapName = "kubeflow-trainer-config"
	var originalUID string
	verifyTrainerReady := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.Phase).To(Equal(common.PhaseReady))

		// Verify the kubeflow-trainer-config ConfigMap exists
		cm, err := k8sClient.CoreV1().ConfigMaps(trainerNamespace).Get(
			ctx, expectedConfigMapName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(),
			"kubeflow-trainer-config ConfigMap should exist")
		g.Expect(cm.Labels).To(HaveKeyWithValue(platformPartOf, trainerPartOf),
			"ConfigMap should have correct labels")
		originalUID = string(cm.UID)
	}
	g.Eventually(verifyTrainerReady).WithTimeout(5 * time.Minute).Should(Succeed())

	t.Logf("Trainer Ready with ConfigMap: %s (UID: %s)", expectedConfigMapName, originalUID)

	// Delete the ConfigMap to simulate drift
	err = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Delete(
		ctx, expectedConfigMapName, metav1.DeleteOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to delete ConfigMap")

	t.Logf("Deleted ConfigMap %s to test drift correction", expectedConfigMapName)

	// Verify ConfigMap is recreated by the controller (drift correction)
	verifyConfigMapRecreated := func(g Gomega) {
		cm, err := k8sClient.CoreV1().ConfigMaps(trainerNamespace).Get(
			ctx, expectedConfigMapName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "ConfigMap should be recreated")
		g.Expect(cm.Name).To(Equal(expectedConfigMapName))
		g.Expect(cm.Labels).To(HaveKeyWithValue(platformPartOf, trainerPartOf),
			"Recreated ConfigMap should have correct labels")
		// Verify it's a new instance (different UID)
		g.Expect(string(cm.UID)).NotTo(Equal(originalUID),
			"Recreated ConfigMap should have a different UID")
	}
	g.Eventually(verifyConfigMapRecreated).WithTimeout(driftCorrectionTimeout).Should(Succeed())

	t.Logf("Drift correction successful: ConfigMap %s was automatically recreated", expectedConfigMapName)
}
