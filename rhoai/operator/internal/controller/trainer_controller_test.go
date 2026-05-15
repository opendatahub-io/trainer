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

package controller

import (
	"context"
	"testing"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
)

const testTrainerName = "default-trainer"

func TestReconcileManaged(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			ManagementState: common.Managed,
			AppNamespace:    "test-trainer-ns",
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, "test-trainer-ns")
	})

	reconciler := newTestReconciler()

	_, err := reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(controllerutil.ContainsFinalizer(updated, finalizerName)).To(BeTrue())

	_, err = reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated = getTrainer(ctx, g)
	g.Expect(updated.Status.ObservedGeneration).To(Equal(updated.Generation))
	g.Expect(updated.Status.Phase).To(Equal(common.PhaseReady))

	readyCond := findCondition(updated, common.ConditionTypeReady)
	g.Expect(readyCond).NotTo(BeNil())
	g.Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))

	provCond := findCondition(updated, common.ConditionTypeProvisioningSucceeded)
	g.Expect(provCond).NotTo(BeNil())
	g.Expect(provCond.Status).To(Equal(metav1.ConditionTrue))
	g.Expect(provCond.Reason).To(Equal("Provisioned"))

	ns := &corev1.Namespace{}
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "test-trainer-ns"}, ns)).To(Succeed())
	g.Expect(ns.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", trainerPartOf))

	cm := &corev1.ConfigMap{}
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "trainer-test-config", Namespace: "default"}, cm)).To(Succeed())
	g.Expect(cm.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", trainerPartOf))
}

func TestReconcileRemoved(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			ManagementState: common.Removed,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() { cleanupTrainer(ctx) })

	reconciler := newTestReconciler()

	_, err := reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	_, err = reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(updated.Status.Phase).To(Equal(common.PhaseNotReady))

	readyCond := findCondition(updated, common.ConditionTypeReady)
	g.Expect(readyCond).NotTo(BeNil())
	g.Expect(readyCond.Status).To(Equal(metav1.ConditionFalse))

	provCond := findCondition(updated, common.ConditionTypeProvisioningSucceeded)
	g.Expect(provCond).NotTo(BeNil())
	g.Expect(provCond.Status).To(Equal(metav1.ConditionFalse))
	g.Expect(provCond.Reason).To(Equal("Removed"))
}

func TestReconcileDelete(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			ManagementState: common.Managed,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())

	reconciler := newTestReconciler()

	_, err := reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(controllerutil.ContainsFinalizer(updated, finalizerName)).To(BeTrue())

	g.Expect(k8sClient.Delete(ctx, updated)).To(Succeed())

	_, err = reconciler.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	deleted := &componentsv1alpha1.Trainer{}
	err = k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, deleted)
	g.Expect(errors.IsNotFound(err)).To(BeTrue())
}

func TestReconcileNotFound(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	reconciler := newTestReconciler()

	result, err := reconciler.Reconcile(ctx, reconcile.Request{
		NamespacedName: types.NamespacedName{Name: "nonexistent"},
	})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(result).To(Equal(reconcile.Result{}))
}

func TestSingletonNameValidation(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wrong-name",
		},
	}
	err := k8sClient.Create(ctx, trainer)
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("default-trainer"))
}

func TestResolveNamespace(t *testing.T) {
	g := NewWithT(t)

	trainer := &componentsv1alpha1.Trainer{
		Spec: componentsv1alpha1.TrainerSpec{AppNamespace: "from-spec"},
	}
	g.Expect(resolveNamespace(trainer)).To(Equal("from-spec"))

	trainer = &componentsv1alpha1.Trainer{}
	g.Expect(resolveNamespace(trainer)).To(Equal(defaultNamespace))
}

func newTestReconciler() *TrainerReconciler {
	return &TrainerReconciler{
		Client:          k8sClient,
		Scheme:          k8sClient.Scheme(),
		ManifestsPath:   testManifestsPath,
		DynamicClient:   dynamicClient,
		DiscoveryClient: discoveryClient,
	}
}

func testRequest() reconcile.Request {
	return reconcile.Request{
		NamespacedName: types.NamespacedName{Name: testTrainerName},
	}
}

func getTrainer(ctx context.Context, g Gomega) *componentsv1alpha1.Trainer {
	trainer := &componentsv1alpha1.Trainer{}
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, trainer)).To(Succeed())
	return trainer
}

func findCondition(trainer *componentsv1alpha1.Trainer, condType common.ConditionType) *common.Condition {
	for i := range trainer.Status.Conditions {
		if trainer.Status.Conditions[i].Type == string(condType) {
			return &trainer.Status.Conditions[i]
		}
	}
	return nil
}

func cleanupTrainer(ctx context.Context) {
	resource := &componentsv1alpha1.Trainer{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, resource); err == nil {
		controllerutil.RemoveFinalizer(resource, finalizerName)
		_ = k8sClient.Update(ctx, resource)
		_ = k8sClient.Delete(ctx, resource)
	}
}

func cleanupNamespace(ctx context.Context, name string) {
	ns := &corev1.Namespace{}
	if err := k8sClient.Get(ctx, client.ObjectKey{Name: name}, ns); err == nil {
		_ = k8sClient.Delete(ctx, ns)
	}
}
