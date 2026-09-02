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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
)

const (
	platformConfigMapName      = "odh-trainer-config"
	platformVersionKey         = "platformVersion"
	initialPlatformVersion     = "2.30.0"
	bumpedPlatformVersion      = "2.31.0"
	platformConfigWatchTimeout = 90 * time.Second
)

// TestPlatformConfigWatch verifies that the operator reacts to platform
// config ConfigMap changes without the Trainer CR being touched (RHOAIENG-88549).
// Before the fix, ConfigMap events were filtered out of the informer cache by
// the part-of=trainer label selector, so status.releases[platform].version
// stayed stale until something else triggered a reconcile.
func TestPlatformConfigWatch(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	if _, err := k8sClient.GetTrainer(ctx); errors.IsNotFound(err) {
		err := k8sClient.CreateTrainer(ctx, trainerNamespace)
		g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	}

	verifyReady := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.Phase).To(Equal(string(common.PhaseReady)))
	}
	g.Eventually(verifyReady).Should(Succeed(), "Trainer should be Ready before testing platform config watch")

	// Set up the platform config ConfigMap without the trainer label,
	// matching how the platform operator creates it in production. If the
	// ConfigMap already exists (e.g. a long-lived cluster where the platform
	// operator owns it), snapshot its data and labels and restore them in
	// cleanup instead of deleting the object.
	var snapshot *corev1.ConfigMap
	cm, err := k8sClient.CoreV1().ConfigMaps(trainerNamespace).Get(ctx, platformConfigMapName, metav1.GetOptions{})
	if err != nil {
		g.Expect(errors.IsNotFound(err)).To(BeTrue(), "unexpected error getting platform config ConfigMap")
		cm = &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      platformConfigMapName,
				Namespace: trainerNamespace,
			},
			Data: map[string]string{
				platformVersionKey: initialPlatformVersion,
			},
		}
		_, err = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Create(ctx, cm, metav1.CreateOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "Failed to create platform config ConfigMap")
	} else {
		snapshot = cm.DeepCopy()
		cm.Data = map[string]string{
			platformVersionKey: initialPlatformVersion,
		}
		cm.Labels = nil
		_, err = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Update(ctx, cm, metav1.UpdateOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "Failed to update pre-existing platform config ConfigMap")
	}
	t.Cleanup(func() {
		if snapshot == nil {
			err := k8sClient.CoreV1().ConfigMaps(trainerNamespace).Delete(ctx, platformConfigMapName, metav1.DeleteOptions{})
			if err != nil {
				g.Expect(errors.IsNotFound(err)).To(BeTrue(), "unexpected error deleting platform config ConfigMap")
			}
			return
		}
		restored, restoreErr := k8sClient.CoreV1().ConfigMaps(trainerNamespace).
			Get(ctx, platformConfigMapName, metav1.GetOptions{})
		if errors.IsNotFound(restoreErr) {
			newCM := snapshot.DeepCopy()
			newCM.ResourceVersion = ""
			newCM.UID = ""
			newCM.CreationTimestamp = metav1.Time{}
			_, restoreErr = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Create(ctx, newCM, metav1.CreateOptions{})
			g.Expect(restoreErr).NotTo(HaveOccurred(), "Failed to restore platform config ConfigMap")
			return
		}
		g.Expect(restoreErr).NotTo(HaveOccurred(), "Failed to get platform config ConfigMap for restore")
		restored.Data = snapshot.Data
		restored.Labels = snapshot.Labels
		_, restoreErr = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Update(ctx, restored, metav1.UpdateOptions{})
		g.Expect(restoreErr).NotTo(HaveOccurred(), "Failed to restore platform config ConfigMap")
	})

	// The initial version must propagate to status.releases.
	g.Eventually(func(g Gomega) { verifyPlatformVersion(g, initialPlatformVersion) }).
		Should(Succeed(), "initial platform version should appear in status.releases")

	// Bump the platform version in the ConfigMap without touching the
	// Trainer CR; the operator must pick up the change on its own.
	cm, err = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Get(ctx, platformConfigMapName, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to get platform config ConfigMap")
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[platformVersionKey] = bumpedPlatformVersion
	_, err = k8sClient.CoreV1().ConfigMaps(trainerNamespace).Update(ctx, cm, metav1.UpdateOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to update platform config ConfigMap")

	// Bounded window: the ConfigMap watch fires within seconds. With a
	// broken watch the status would only update if some unrelated watched
	// resource happened to enqueue a reconcile, which does not happen on a
	// quiet e2e cluster within this window.
	g.Eventually(func(g Gomega) { verifyPlatformVersion(g, bumpedPlatformVersion) }).
		WithTimeout(platformConfigWatchTimeout).
		Should(Succeed(), "ConfigMap change should update status.releases without touching the Trainer CR")
}

func verifyPlatformVersion(g Gomega, version string) {
	trainer, err := k8sClient.GetTrainer(ctx)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(platformReleaseVersion(trainer)).To(Equal(version),
		"status.releases should carry the platform version from the ConfigMap")
}

func platformReleaseVersion(trainer *componentsv1alpha1.Trainer) string {
	for i := range trainer.Status.Releases {
		if trainer.Status.Releases[i].Name == "platform" {
			return trainer.Status.Releases[i].Version
		}
	}
	return ""
}
