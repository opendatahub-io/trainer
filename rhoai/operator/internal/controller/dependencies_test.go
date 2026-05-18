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
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/kubernetes/scheme"
	k8stesting "k8s.io/client-go/testing"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestCheckJobSetAvailable(t *testing.T) {
	ctx := context.Background()

	t.Run("returns true when JobSet CRD exists and is established", func(t *testing.T) {
		g := NewWithT(t)

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

		// Create fake discovery client
		fakeDiscovery := &fakediscovery.FakeDiscovery{
			Fake: &k8stesting.Fake{},
		}
		fakeDiscovery.Resources = []*metav1.APIResourceList{
			{
				GroupVersion: "jobset.x-k8s.io/v1alpha2",
				APIResources: []metav1.APIResource{
					{
						Name: jobSetResourceName,
						Kind: "JobSet",
					},
				},
			},
		}

		reconciler := &TrainerReconciler{
			Client:          fakeClient,
			DiscoveryClient: fakeDiscovery,
		}

		available := reconciler.checkJobSetAvailable(ctx)
		g.Expect(available).To(BeTrue())
	})

	t.Run("returns false when JobSet CRD does not exist", func(t *testing.T) {
		g := NewWithT(t)

		// Create scheme without JobSet CRD
		s := runtime.NewScheme()
		_ = apiextensionsv1.AddToScheme(s)

		// Create fake client without the CRD
		fakeClient := fake.NewClientBuilder().
			WithScheme(s).
			Build()

		// Create fake discovery client with empty resources
		fakeDiscovery := &fakediscovery.FakeDiscovery{
			Fake: &k8stesting.Fake{},
		}
		fakeDiscovery.Resources = []*metav1.APIResourceList{}

		reconciler := &TrainerReconciler{
			Client:          fakeClient,
			DiscoveryClient: fakeDiscovery,
		}

		available := reconciler.checkJobSetAvailable(ctx)
		g.Expect(available).To(BeFalse())
	})

	t.Run("returns false when JobSet CRD exists but not established", func(t *testing.T) {
		g := NewWithT(t)

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

		// Create fake discovery client with empty resources
		fakeDiscovery := &fakediscovery.FakeDiscovery{
			Fake: &k8stesting.Fake{},
		}

		reconciler := &TrainerReconciler{
			Client:          fakeClient,
			DiscoveryClient: fakeDiscovery,
		}

		available := reconciler.checkJobSetAvailable(ctx)
		g.Expect(available).To(BeFalse())
	})
}

func TestGetJobSetMissingMessage(t *testing.T) {
	g := NewWithT(t)

	message := getJobSetMissingMessage()
	g.Expect(message).To(ContainSubstring(jobSetCRDName))
	g.Expect(message).To(ContainSubstring(jobSetVersion))
	g.Expect(message).To(ContainSubstring("JobSet Operator"))
	g.Expect(message).To(ContainSubstring("TrainJob"))
}

func init() {
	// Register CRD types with the default scheme for tests
	_ = apiextensionsv1.AddToScheme(scheme.Scheme)
}
