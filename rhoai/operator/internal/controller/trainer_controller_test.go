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
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
)

func TestTrainerControllerReconcile(t *testing.T) {
	g := NewWithT(t)

	const resourceName = "default-trainer"

	ctx := context.Background()

	typeNamespacedName := types.NamespacedName{
		Name: resourceName,
	}

	// Create the custom resource for the Kind Trainer
	trainer := &componentsv1alpha1.Trainer{}
	err := k8sClient.Get(ctx, typeNamespacedName, trainer)
	if err != nil && errors.IsNotFound(err) {
		resource := &componentsv1alpha1.Trainer{
			ObjectMeta: metav1.ObjectMeta{
				Name: resourceName,
			},
		}
		g.Expect(k8sClient.Create(ctx, resource)).To(Succeed())
	}

	t.Cleanup(func() {
		resource := &componentsv1alpha1.Trainer{}
		err := k8sClient.Get(ctx, typeNamespacedName, resource)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
	})

	// Reconcile the created resource
	controllerReconciler := &TrainerReconciler{
		Client: k8sClient,
		Scheme: k8sClient.Scheme(),
	}

	_, err = controllerReconciler.Reconcile(ctx, reconcile.Request{
		NamespacedName: typeNamespacedName,
	})
	g.Expect(err).NotTo(HaveOccurred())
}
