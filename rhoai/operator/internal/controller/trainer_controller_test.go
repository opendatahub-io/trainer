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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
)

const (
	testTrainerName      = "default-trainer"
	testTrainerNamespace = "test-trainer-ns"
	testCRDSchemaType    = "object"
)

func TestReconcileManaged(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create JobSet CRD to satisfy CRD dependency check
	jobSetCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "jobset.x-k8s.io",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   "JobSet",
				Plural: "jobsets",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    jobSetVersion,
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: testCRDSchemaType,
						},
					},
				},
			},
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{
			Conditions: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:   apiextensionsv1.Established,
					Status: apiextensionsv1.ConditionTrue,
				},
			},
		},
	}
	g.Expect(k8sClient.Create(ctx, jobSetCRD)).To(Succeed())
	t.Cleanup(func() {
		_ = k8sClient.Delete(ctx, jobSetCRD)
	})

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			ManagementState: common.Managed,
			AppNamespace:    testTrainerNamespace,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, testTrainerNamespace)
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
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerNamespace}, ns)).To(Succeed())
	g.Expect(ns.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", trainerPartOf))

	cm := &corev1.ConfigMap{}
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "trainer-test-config", Namespace: testTrainerNamespace}, cm)).To(Succeed())
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

func TestCleanupTrainerResourcesDeletesLabeledResources(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	ctrGVR := schema.GroupVersionResource{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "clustertrainingruntimes"}
	ctrGVK := schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: "ClusterTrainingRuntime"}
	trGVR := schema.GroupVersionResource{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "trainingruntimes"}
	trGVK := schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: "TrainingRuntime"}

	labeledCTR := newUnstructured(ctrGVK, "labeled-ctr", "", map[string]string{"platform.opendatahub.io/part-of": trainerPartOf})
	_, err := dynamicClient.Resource(ctrGVR).Create(ctx, labeledCTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	unlabeledCTR := newUnstructured(ctrGVK, "unlabeled-ctr", "", nil)
	_, err = dynamicClient.Resource(ctrGVR).Create(ctx, unlabeledCTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "tr-test-ns"}}
	g.Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	t.Cleanup(func() { _ = k8sClient.Delete(ctx, ns) })

	labeledTR := newUnstructured(trGVK, "labeled-tr", "tr-test-ns", map[string]string{"platform.opendatahub.io/part-of": trainerPartOf})
	_, err = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Create(ctx, labeledTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	t.Cleanup(func() {
		_ = dynamicClient.Resource(ctrGVR).Delete(ctx, "labeled-ctr", metav1.DeleteOptions{})
		_ = dynamicClient.Resource(ctrGVR).Delete(ctx, "unlabeled-ctr", metav1.DeleteOptions{})
		_ = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Delete(ctx, "labeled-tr", metav1.DeleteOptions{})
	})

	reconciler := newTestReconciler()
	reconciler.cleanupTrainerResources(ctx)

	_, err = dynamicClient.Resource(ctrGVR).Get(ctx, "labeled-ctr", metav1.GetOptions{})
	g.Expect(errors.IsNotFound(err)).To(BeTrue(), "labeled CTR should be deleted")

	_, err = dynamicClient.Resource(ctrGVR).Get(ctx, "unlabeled-ctr", metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "unlabeled CTR should remain")

	_, err = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Get(ctx, "labeled-tr", metav1.GetOptions{})
	g.Expect(errors.IsNotFound(err)).To(BeTrue(), "labeled TrainingRuntime should be deleted")
}

func newUnstructured(gvk schema.GroupVersionKind, name, namespace string, labels map[string]string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)
	obj.SetName(name)
	if namespace != "" {
		obj.SetNamespace(namespace)
	}
	if labels != nil {
		obj.SetLabels(labels)
	}
	return obj
}

func newTestReconciler() *TrainerReconciler {
	return &TrainerReconciler{
		Client:           k8sClient,
		Scheme:           k8sClient.Scheme(),
		ManifestsPath:    testManifestsPath,
		ImageStreamsPath: testImageStreamsPath,
		WorkDir:          testWorkDir,
		DynamicClient:    dynamicClient,
		DiscoveryClient:  discoveryClient,
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
