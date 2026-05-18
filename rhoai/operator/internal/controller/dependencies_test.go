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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	available := reconciler.checkJobSetAvailable(ctx)
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

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	available := reconciler.checkJobSetAvailable(ctx)
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

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	available := reconciler.checkJobSetAvailable(ctx)
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

func TestCheckJobSetOperatorCRWhenCRDDoesNotExist(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create scheme without JobSetOperator CRD
	s := runtime.NewScheme()
	_ = apiextensionsv1.AddToScheme(s)

	// Create fake client without the JobSetOperator CRD
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		Build()

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	// Should return true (skip check) when CRD doesn't exist
	available := reconciler.checkJobSetOperatorCR(ctx)
	g.Expect(available).To(BeTrue())
}

func TestCheckJobSetOperatorCRWhenCRDExistsButCRDoesNotExist(t *testing.T) {
	g := NewWithT(t)
	ctx := context.Background()

	// Create JobSetOperator CRD
	jobSetOperatorCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "jobsetoperators.operator.openshift.io",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "operator.openshift.io",
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   "JobSetOperator",
				Plural: "jobsetoperators",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
						},
					},
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

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	// Should return false when CRD exists but CR doesn't
	available := reconciler.checkJobSetOperatorCR(ctx)
	g.Expect(available).To(BeFalse())
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

	reconciler := &TrainerReconciler{
		Client: fakeClient,
	}

	installed := reconciler.checkJobSetOperatorInstalled(ctx)
	g.Expect(installed).To(BeFalse())
}

func TestGetJobSetOperatorNotInstalledMessage(t *testing.T) {
	g := NewWithT(t)

	message := getJobSetOperatorNotInstalledMessage()
	g.Expect(message).To(ContainSubstring("JobSet Operator"))
	g.Expect(message).To(ContainSubstring("not installed"))
	g.Expect(message).To(ContainSubstring("OLM"))
}

func init() {
	// Register CRD types with the default scheme for tests
	_ = apiextensionsv1.AddToScheme(scheme.Scheme)
}
