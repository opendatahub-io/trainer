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
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"

	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/actions"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/reconciler"
	fwtypes "github.com/opendatahub-io/odh-platform-utilities/framework/controller/types"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
)

const (
	testTrainerName      = "default-trainer"
	testTrainerNamespace = "test-trainer-ns"
	testCRDSchemaType    = "object"
	testJobSetPlural     = "jobsets"
)

func TestReconcileManaged(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	createJobSetCRD(ctx, t, g)

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			AppNamespace: testTrainerNamespace,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, testTrainerNamespace)
	})

	r := newTestReconciler(t)

	_, err := r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(controllerutil.ContainsFinalizer(updated, finalizerName)).To(BeTrue())
	g.Expect(updated.Status.Phase).To(Equal(common.PhaseReady))

	readyCond := findCondition(updated, common.ConditionTypeReady)
	g.Expect(readyCond).NotTo(BeNil())
	g.Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))

	provCond := findCondition(updated, common.ConditionTypeProvisioningSucceeded)
	g.Expect(provCond).NotTo(BeNil())
	g.Expect(provCond.Status).To(Equal(metav1.ConditionTrue))

	ns := &corev1.Namespace{}
	g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerNamespace}, ns)).To(Succeed())
	g.Expect(ns.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", trainerPartOf))
}

func TestReconcileDelete(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	createJobSetCRD(ctx, t, g)

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() { cleanupTrainer(ctx) })

	r := newTestReconciler(t)

	_, err := r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(controllerutil.ContainsFinalizer(updated, finalizerName)).To(BeTrue())

	g.Expect(k8sClient.Delete(ctx, updated)).To(Succeed())

	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	deleted := &componentsv1alpha1.Trainer{}
	err = k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, deleted)
	g.Expect(errors.IsNotFound(err)).To(BeTrue())
}

func TestReconcileNotFound(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	r := newTestReconciler(t)

	result, err := r.Reconcile(ctx, reconcile.Request{
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
	ctrGVK := schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: clusterTrainingRuntime}
	trGVR := schema.GroupVersionResource{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "trainingruntimes"}
	trGVK := schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: "TrainingRuntime"}

	labeledCTR := newUnstructured(ctrGVK, "labeled-ctr", "", map[string]string{labels.PlatformPartOf: trainerPartOf})
	_, err := dynamicClient.Resource(ctrGVR).Create(ctx, labeledCTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	unlabeledCTR := newUnstructured(ctrGVK, "unlabeled-ctr", "", nil)
	_, err = dynamicClient.Resource(ctrGVR).Create(ctx, unlabeledCTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "tr-test-ns"}}
	g.Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	t.Cleanup(func() { _ = k8sClient.Delete(ctx, ns) })

	labeledTR := newUnstructured(trGVK, "labeled-tr", "tr-test-ns", map[string]string{labels.PlatformPartOf: trainerPartOf})
	_, err = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Create(ctx, labeledTR, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred())

	t.Cleanup(func() {
		_ = dynamicClient.Resource(ctrGVR).Delete(ctx, "labeled-ctr", metav1.DeleteOptions{})
		_ = dynamicClient.Resource(ctrGVR).Delete(ctx, "unlabeled-ctr", metav1.DeleteOptions{})
		_ = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Delete(ctx, "labeled-tr", metav1.DeleteOptions{})
	})

	cleanupTrainerResources(ctx, dynamicClient)

	_, err = dynamicClient.Resource(ctrGVR).Get(ctx, "labeled-ctr", metav1.GetOptions{})
	g.Expect(errors.IsNotFound(err)).To(BeTrue(), "labeled CTR should be deleted")

	_, err = dynamicClient.Resource(ctrGVR).Get(ctx, "unlabeled-ctr", metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "unlabeled CTR should remain")

	_, err = dynamicClient.Resource(trGVR).Namespace("tr-test-ns").Get(ctx, "labeled-tr", metav1.GetOptions{})
	g.Expect(errors.IsNotFound(err)).To(BeTrue(), "labeled TrainingRuntime should be deleted")
}

func TestGetComponentReleases(t *testing.T) {
	g := NewWithT(t)

	m := newTestActions()

	releases, err := m.getComponentReleases()
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(releases).NotTo(BeEmpty())

	g.Expect(releases[0].Name).To(Equal("Kubeflow Trainer"))
	g.Expect(releases[0].Version).To(Equal("2.1.0"))
	g.Expect(releases[0].RepoURL).NotTo(BeEmpty())
}

func TestReconcileManagedPopulatesReleases(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	createJobSetCRD(ctx, t, g)

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			AppNamespace: testTrainerNamespace,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, testTrainerNamespace)
	})

	r := newTestReconciler(t)

	_, err := r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	g.Expect(updated.Status.Releases).NotTo(BeEmpty(), "Releases array should be populated")
	g.Expect(updated.Status.Releases[0].Name).To(Equal("Kubeflow Trainer"))
	g.Expect(updated.Status.Releases[0].Version).To(Equal("2.1.0"))
	g.Expect(updated.Status.Releases[0].RepoURL).To(Equal("https://github.com/kubeflow/trainer"))
}

func TestPlatformVersionHandshake(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	createJobSetCRD(ctx, t, g)

	handshakeNS := "handshake-test-ns"
	platformVersion := "2.20.0"

	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			AppNamespace: handshakeNS,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, handshakeNS)
	})

	r := newTestReconciler(t)

	// First reconcile: no platform ConfigMap exists
	_, err := r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated := getTrainer(ctx, g)
	for _, rel := range updated.Status.Releases {
		g.Expect(rel.Name).NotTo(Equal(platformReleaseName), "platform release should not exist without ConfigMap")
	}

	// Create the platform config ConfigMap
	platformCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      platformConfigMapName,
			Namespace: handshakeNS,
			Labels: map[string]string{
				labels.PlatformPartOf: trainerPartOf,
			},
		},
		Data: map[string]string{
			platformVersionKey: platformVersion,
		},
	}
	g.Expect(k8sClient.Create(ctx, platformCM)).To(Succeed())
	t.Cleanup(func() { _ = k8sClient.Delete(ctx, platformCM) })

	// Reconcile again — platform version should appear in releases
	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated = getTrainer(ctx, g)
	var platformRelease *common.ComponentRelease
	for i, rel := range updated.Status.Releases {
		if rel.Name == platformReleaseName {
			platformRelease = &updated.Status.Releases[i]
			break
		}
	}
	g.Expect(platformRelease).NotTo(BeNil(), "platform release entry should exist")
	g.Expect(platformRelease.Version).To(Equal(platformVersion))

	// Simulate platform upgrade
	upgradedVersion := "2.21.0"
	platformCM.Data[platformVersionKey] = upgradedVersion
	g.Expect(k8sClient.Update(ctx, platformCM)).To(Succeed())

	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated = getTrainer(ctx, g)
	platformRelease = nil
	for i, rel := range updated.Status.Releases {
		if rel.Name == platformReleaseName {
			platformRelease = &updated.Status.Releases[i]
			break
		}
	}
	g.Expect(platformRelease).NotTo(BeNil(), "platform release should exist after upgrade")
	g.Expect(platformRelease.Version).To(Equal(upgradedVersion))
}

func TestReconcileDeleteAndRecreate(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	createJobSetCRD(ctx, t, g)

	lifecycleNS := "lifecycle-test-ns"
	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			AppNamespace: lifecycleNS,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())
	t.Cleanup(func() {
		cleanupTrainer(ctx)
		cleanupNamespace(ctx, lifecycleNS)
	})

	r := newTestReconciler(t)

	_, err := r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(getTrainer(ctx, g).Status.Phase).To(Equal(common.PhaseReady))

	// Delete the CR
	updated := getTrainer(ctx, g)
	g.Expect(k8sClient.Delete(ctx, updated)).To(Succeed())

	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	err = k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, &componentsv1alpha1.Trainer{})
	g.Expect(errors.IsNotFound(err)).To(BeTrue(), "Trainer should be deleted")

	// Recreate the CR
	trainer = &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: testTrainerName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			AppNamespace: lifecycleNS,
		},
	}
	g.Expect(k8sClient.Create(ctx, trainer)).To(Succeed())

	// First reconcile: adds finalizer + runs pipeline. The addFinalizer Update
	// changes the resource version, so a second reconcile is needed for the
	// framework's status SSA to apply cleanly.
	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	_, err = r.Reconcile(ctx, testRequest())
	g.Expect(err).NotTo(HaveOccurred())

	updated = getTrainer(ctx, g)
	g.Expect(updated.Status.Phase).To(Equal(common.PhaseReady))
}

func TestEnsureNamespaceAlreadyExists(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	testNS := "preexisting-ns"
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: testNS},
	}
	g.Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	t.Cleanup(func() { cleanupNamespace(ctx, testNS) })

	m := newTestActions()
	err := m.ensureNamespace(ctx, &fwtypes.ReconciliationRequest{
		Client:   k8sClient,
		Instance: &componentsv1alpha1.Trainer{Spec: componentsv1alpha1.TrainerSpec{AppNamespace: testNS}},
	})
	g.Expect(err).NotTo(HaveOccurred())
}

func TestIsImmutableFieldError(t *testing.T) {
	g := NewWithT(t)

	g.Expect(isImmutableFieldError(fmt.Errorf("field is immutable"))).To(BeTrue())
	g.Expect(isImmutableFieldError(fmt.Errorf("apply failed: field is immutable; 2 more failed"))).To(BeTrue())
	g.Expect(isImmutableFieldError(fmt.Errorf("not found"))).To(BeFalse())
	g.Expect(isImmutableFieldError(nil)).To(BeFalse())
}

// --- Helpers ---

func newTestReconciler(t *testing.T) *reconciler.Reconciler {
	t.Helper()

	m := newTestActions()
	// Only register ProvisioningSucceeded as dependent — the test pipeline
	// omits deployments.NewAction, so DeploymentsAvailable would never be set
	// and CleanupStaleConditions would mark it as an error.
	r, err := reconciler.NewReconciler(testMgr, "trainer", &componentsv1alpha1.Trainer{},
		reconciler.WithFinalizerName(finalizerName),
		reconciler.WithConditionsManagerFactory("Ready", "ProvisioningSucceeded"),
	)
	if err != nil {
		t.Fatalf("failed to create reconciler: %v", err)
	}

	r.Client = k8sClient
	m.reconciler = r

	r.Actions = []actions.Fn{
		m.checkDependencies,
		m.ensureNamespace,
		m.updateReleases,
		m.renderManifests,
	}
	r.Finalizer = []actions.Fn{m.cleanup}

	return r
}

func newTestActions() *trainerActions {
	return &trainerActions{
		apiReader:        k8sClient,
		manifestsPath:    testManifestsPath,
		runtimesPath:     testRuntimesPath,
		imageStreamsPath: testImageStreamsPath,
		workDir:          testWorkDir,
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
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, resource); err != nil {
		return
	}

	controllerutil.RemoveFinalizer(resource, finalizerName)
	_ = k8sClient.Update(ctx, resource)
	_ = k8sClient.Delete(ctx, resource)

	_ = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		err := k8sClient.Get(ctx, types.NamespacedName{Name: testTrainerName}, &componentsv1alpha1.Trainer{})
		return errors.IsNotFound(err), nil
	})
}

func cleanupNamespace(ctx context.Context, name string) {
	ns := &corev1.Namespace{}
	if err := k8sClient.Get(ctx, client.ObjectKey{Name: name}, ns); err == nil {
		_ = k8sClient.Delete(ctx, ns)
	}
}

func createJobSetCRD(ctx context.Context, t *testing.T, g Gomega) {
	t.Helper()

	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: jobSetGroup,
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   jobSetKind,
				Plural: testJobSetPlural,
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
	g.Expect(k8sClient.Create(ctx, crd)).To(Succeed())
	t.Cleanup(func() {
		deleteJobSetCRDAndWait(ctx)
	})
}

func deleteJobSetCRDAndWait(ctx context.Context) {
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: jobSetCRDName}, crd); err == nil {
		_ = k8sClient.Delete(ctx, crd)
	}
	_ = wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		err := k8sClient.Get(ctx, types.NamespacedName{Name: jobSetCRDName}, &apiextensionsv1.CustomResourceDefinition{})
		return errors.IsNotFound(err), nil
	})
}

func newUnstructured(gvk schema.GroupVersionKind, name, namespace string, objLabels map[string]string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)
	obj.SetName(name)
	if namespace != "" {
		obj.SetNamespace(namespace)
	}
	if objLabels != nil {
		obj.SetLabels(objLabels)
	}
	return obj
}
