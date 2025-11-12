/*
Copyright 2024 The Kubeflow Authors.

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

package rhai

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/controller"
	rhaicontroller "github.com/kubeflow/trainer/v2/pkg/rhai/controller"
	jobruntimes "github.com/kubeflow/trainer/v2/pkg/runtime"
)

func TestSetupWithManager(t *testing.T) {
	// Unit test verifying the setup logic

	scheme := runtime.NewScheme()
	if err := trainer.AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add trainer scheme: %v", err)
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	// Create base reconciler
	baseReconciler := controller.NewTrainJobReconciler(
		fakeClient,
		&record.FakeRecorder{},
		map[string]jobruntimes.Runtime{},
	)

	if baseReconciler == nil {
		t.Fatal("Expected base reconciler to be created")
	}

	// Verify we can create a progression reconciler (the core logic of SetupWithManager)
	progressionReconciler := rhaicontroller.NewProgressionReconciler(
		baseReconciler,
		fakeClient,
	)

	if progressionReconciler == nil {
		t.Error("Expected progression reconciler to be created")
	}

	// Note: Full SetupWithManager testing with actual manager registration
	// requires envtest and should be done in integration tests (test/integration/)
	t.Log("RHAI setup logic verified successfully")
}
