package controller

import (
	"testing"

	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"
)

func TestIsImmutableFieldError(t *testing.T) {
	g := NewWithT(t)

	immutableErr := k8serr.NewInvalid(
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		"test-deploy",
		field.ErrorList{
			field.Invalid(field.NewPath("spec", "selector"), nil, "field is immutable"),
		},
	)

	g.Expect(isImmutableFieldError(immutableErr)).To(BeTrue())
	g.Expect(isImmutableFieldError(k8serr.NewNotFound(schema.GroupResource{}, "x"))).To(BeFalse())
	g.Expect(isImmutableFieldError(k8serr.NewConflict(schema.GroupResource{}, "x", nil))).To(BeFalse())
}

func TestApplyResourcesRecreatesDeploymentOnImmutableSelector(t *testing.T) {
	g := NewWithT(t)

	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "test-immutable-selector"},
	}
	g.Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	t.Cleanup(func() {
		_ = k8sClient.Delete(ctx, ns)
	})

	oldLabels := map[string]string{"app": trainerPartOf}
	deploy := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deploy",
			Namespace: ns.Name,
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: oldLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: oldLabels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "manager",
						Image: "test:v1",
					}},
				},
			},
		},
	}
	g.Expect(k8sClient.Create(ctx, deploy)).To(Succeed())

	newLabels := map[string]string{"app": trainerPartOf, labels.PlatformPartOf: trainerPartOf}
	newDeploy := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deploy",
			Namespace: ns.Name,
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: newLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: newLabels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "manager",
						Image: "test:v2",
					}},
				},
			},
		},
	}

	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(newDeploy)
	g.Expect(err).NotTo(HaveOccurred())

	rendered := []unstructured.Unstructured{{Object: u}}
	rendered[0].SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("Deployment"))

	// First call: deletes old Deployment and returns error to trigger requeue
	err = applyResources(ctx, k8sClient, rendered)
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("immutable field change"))

	// Verify old Deployment was deleted
	g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(newDeploy), &appsv1.Deployment{})).
		To(MatchError(ContainSubstring("not found")))

	// Second call (simulating next reconcile): creates Deployment with new selector
	u, err = runtime.DefaultUnstructuredConverter.ToUnstructured(newDeploy)
	g.Expect(err).NotTo(HaveOccurred())
	rendered = []unstructured.Unstructured{{Object: u}}
	rendered[0].SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("Deployment"))

	g.Expect(applyResources(ctx, k8sClient, rendered)).To(Succeed())

	var result appsv1.Deployment
	g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(newDeploy), &result)).To(Succeed())
	g.Expect(result.Spec.Selector.MatchLabels).To(Equal(newLabels))
	g.Expect(result.Spec.Template.Spec.Containers[0].Image).To(Equal("test:v2"))
}

func TestApplyResourcesUsesSSA(t *testing.T) {
	g := NewWithT(t)

	rendered := []unstructured.Unstructured{{
		Object: map[string]any{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]any{
				"name":      "test-ssa-cm",
				"namespace": "default",
			},
			"data": map[string]any{"key": "v1"},
		},
	}}

	g.Expect(applyResources(ctx, k8sClient, rendered)).To(Succeed())

	rendered[0].Object["data"] = map[string]any{"key": "v2"}
	g.Expect(applyResources(ctx, k8sClient, rendered)).To(Succeed())

	var result corev1.ConfigMap
	g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: "test-ssa-cm", Namespace: "default"}, &result)).To(Succeed())
	g.Expect(result.Data["key"]).To(Equal("v2"))

	t.Cleanup(func() {
		_ = k8sClient.Delete(ctx, &result)
	})
}
