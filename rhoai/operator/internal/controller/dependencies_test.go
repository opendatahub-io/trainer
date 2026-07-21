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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testObjectType        = "object"
	jobSetOperatorCRDName = "jobsetoperators.operator.openshift.io"

	condDegraded                              = "Degraded"
	condAvailable                             = "Available"
	condTargetConfigControllerDegraded        = "TargetConfigControllerDegraded"
	condJobSetOperatorStaticResourcesDegraded = "JobSetOperatorStaticResourcesDegraded"

	statusTrue  = "True"
	statusFalse = "False"

	condFieldType   = "type"
	condFieldStatus = "status"
	condFieldReason = "reason"
	pluralJobSetOps = "jobsetoperators"
)

func TestCheckJobSetAvailableWhenCRDExistsAndEstablished(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create a fake CRD for JobSet
	jobSetCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
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

	// Create scheme and add CRD types
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client with the CRD
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(jobSetCRD).
		Build()

	c := fakeClient

	available := checkJobSetAvailable(ctx, c)
	g.Expect(available).To(BeTrue())
}

func TestCheckJobSetAvailableWhenCRDDoesNotExist(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create scheme without JobSet CRD
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client without the CRD
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		Build()

	c := fakeClient

	available := checkJobSetAvailable(ctx, c)
	g.Expect(available).To(BeFalse())
}

func TestCheckJobSetAvailableWhenCRDNotEstablished(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create a fake CRD for JobSet that's not established
	jobSetCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{
			Conditions: []apiextensionsv1.CustomResourceDefinitionCondition{
				{
					Type:   apiextensionsv1.Established,
					Status: apiextensionsv1.ConditionFalse,
				},
			},
		},
	}

	// Create scheme and add CRD types
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client with the CRD
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(jobSetCRD).
		Build()

	c := fakeClient

	available := checkJobSetAvailable(ctx, c)
	g.Expect(available).To(BeFalse())
}

func TestGetJobSetMissingMessage(t *testing.T) {
	g := NewWithT(t)

	message := getJobSetMissingMessage()
	g.Expect(message).To(ContainSubstring(jobSetCRDName))
	g.Expect(message).To(ContainSubstring(jobSetVersion))
	g.Expect(message).To(ContainSubstring("JobSet"))
	g.Expect(message).To(ContainSubstring("CRD"))
}

func TestCheckJobSetOperatorCRWhenCRDExistsButCRDoesNotExist(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create JobSetOperator CRD
	jobSetOperatorCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetOperatorCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: jobSetOperatorGroup,
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   jobSetOperatorKind,
				Plural: pluralJobSetOps,
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    jobSetOperatorAPIVersion,
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: testObjectType,
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

	// Create scheme and add CRD types
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client with CRD but without the CR
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(jobSetOperatorCRD).
		Build()

	c := fakeClient

	// Should return false when CRD exists but CR doesn't
	available := checkJobSetOperatorCR(ctx, c)
	g.Expect(available).To(BeFalse())
}

func TestCheckJobSetOperatorCRWhenCRDAndCRExist(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create JobSetOperator CRD
	jobSetOperatorCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetOperatorCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: jobSetOperatorGroup,
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   jobSetOperatorKind,
				Plural: pluralJobSetOps,
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    jobSetOperatorAPIVersion,
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: testObjectType,
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

	// Create JobSetOperator CR named "cluster"
	jobSetOperatorCR := &unstructured.Unstructured{}
	jobSetOperatorCR.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   jobSetOperatorGroup,
		Version: jobSetOperatorAPIVersion,
		Kind:    jobSetOperatorKind,
	})
	jobSetOperatorCR.SetName(jobSetOperatorCRName)

	// Create scheme and add CRD types
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client with both CRD and CR
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(jobSetOperatorCRD, jobSetOperatorCR).
		Build()

	c := fakeClient

	// Should return true when both CRD and CR exist (happy path)
	available := checkJobSetOperatorCR(ctx, c)
	g.Expect(available).To(BeTrue())
}

func TestGetJobSetOperatorCRMissingMessage(t *testing.T) {
	g := NewWithT(t)

	message := getJobSetOperatorCRMissingMessage()
	g.Expect(message).To(ContainSubstring(jobSetOperatorCRName))
	g.Expect(message).To(ContainSubstring("JobSetOperator CR"))
	g.Expect(message).To(ContainSubstring("cluster"))
}

func TestCheckJobSetOperatorInstalledWhenNoOperatorFound(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create scheme
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client with no operator
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		Build()

	c := fakeClient

	installed := checkJobSetOperatorInstalled(ctx, c)
	g.Expect(installed).To(BeFalse())
}

func TestGetJobSetOperatorNotInstalledMessage(t *testing.T) {
	g := NewWithT(t)

	message := getJobSetOperatorNotInstalledMessage()
	g.Expect(message).To(ContainSubstring("JobSet Operator"))
	g.Expect(message).To(ContainSubstring("not installed"))
	g.Expect(message).To(ContainSubstring("OLM"))
}

func TestCheckJobSetOperatorHealthWhenHealthy(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	cr := newJobSetOperatorCR([]map[string]interface{}{
		{condFieldType: condDegraded, condFieldStatus: statusFalse, condFieldReason: "AsExpected"},
		{condFieldType: condAvailable, condFieldStatus: statusTrue, condFieldReason: "AsExpected"},
	})

	c := newClientWithJobSetOperatorCR(cr)
	healthy, err := checkJobSetOperatorHealth(ctx, c)
	g.Expect(healthy).To(BeTrue())
	g.Expect(err).NotTo(HaveOccurred())
}

func TestCheckJobSetOperatorHealthWhenNoConditions(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	cr := newJobSetOperatorCR(nil)

	c := newClientWithJobSetOperatorCR(cr)
	healthy, err := checkJobSetOperatorHealth(ctx, c)
	g.Expect(healthy).To(BeFalse())
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("no status conditions"))
}

func TestCheckJobSetOperatorHealthWhenDegradedTrue(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	cr := newJobSetOperatorCR([]map[string]interface{}{
		{condFieldType: condDegraded, condFieldStatus: statusTrue, condFieldReason: "SomethingBroke", "message": "controller failed"},
	})

	c := newClientWithJobSetOperatorCR(cr)
	healthy, err := checkJobSetOperatorHealth(ctx, c)
	g.Expect(healthy).To(BeFalse())
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("Degraded=True"))
	g.Expect(err.Error()).To(ContainSubstring("SomethingBroke"))
	g.Expect(err.Error()).To(ContainSubstring("controller failed"))
}

func TestCheckJobSetOperatorHealthWhenAvailableFalse(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	cr := newJobSetOperatorCR([]map[string]interface{}{
		{condFieldType: condAvailable, condFieldStatus: statusFalse, condFieldReason: "NotReady"},
	})

	c := newClientWithJobSetOperatorCR(cr)
	healthy, err := checkJobSetOperatorHealth(ctx, c)
	g.Expect(healthy).To(BeFalse())
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("Available=False"))
}

func TestCheckJobSetOperatorHealthWithMultipleDegradedConditions(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	cr := newJobSetOperatorCR([]map[string]interface{}{
		{condFieldType: condDegraded, condFieldStatus: statusTrue, condFieldReason: "Broken"},
		{condFieldType: condAvailable, condFieldStatus: statusFalse, condFieldReason: "Down"},
	})

	c := newClientWithJobSetOperatorCR(cr)
	healthy, err := checkJobSetOperatorHealth(ctx, c)
	g.Expect(healthy).To(BeFalse())
	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring("Degraded=True"))
	g.Expect(err.Error()).To(ContainSubstring("Available=False"))
}

func TestIsJobSetOperatorConditionDegraded(t *testing.T) {
	tests := []struct {
		condType   string
		condStatus string
		want       bool
	}{
		{condDegraded, statusTrue, true},
		{condDegraded, statusFalse, false},
		{condTargetConfigControllerDegraded, statusTrue, true},
		{condTargetConfigControllerDegraded, statusFalse, false},
		{condJobSetOperatorStaticResourcesDegraded, statusTrue, true},
		{condAvailable, statusFalse, true},
		{condAvailable, statusTrue, false},
		{"Progressing", statusTrue, false},
	}

	for _, tt := range tests {
		t.Run(tt.condType+"="+tt.condStatus, func(t *testing.T) {
			g := NewWithT(t)
			g.Expect(isJobSetOperatorConditionDegraded(tt.condType, tt.condStatus)).To(Equal(tt.want))
		})
	}
}

func newJobSetOperatorCR(conditions []map[string]interface{}) *unstructured.Unstructured {
	cr := &unstructured.Unstructured{}
	cr.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   jobSetOperatorGroup,
		Version: jobSetOperatorAPIVersion,
		Kind:    jobSetOperatorKind,
	})
	cr.SetName(jobSetOperatorCRName)

	if conditions != nil {
		condSlice := make([]interface{}, len(conditions))
		for i, c := range conditions {
			condSlice[i] = c
		}
		_ = unstructured.SetNestedSlice(cr.Object, condSlice, "status", "conditions")
	}

	return cr
}

func newClientWithJobSetOperatorCR(cr *unstructured.Unstructured) client.Client {
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: jobSetOperatorCRDName},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: jobSetOperatorGroup,
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   jobSetOperatorKind,
				Plural: pluralJobSetOps,
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name: jobSetOperatorAPIVersion, Served: true, Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: testObjectType},
					},
				},
			},
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{
			Conditions: []apiextensionsv1.CustomResourceDefinitionCondition{
				{Type: apiextensionsv1.Established, Status: apiextensionsv1.ConditionTrue},
			},
		},
	}

	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	return fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(crd, cr).
		Build()
}

func init() {
	// Register CRD types with the default scheme for tests
	_ = apiextensionsv1.AddToScheme(scheme.Scheme)
}
