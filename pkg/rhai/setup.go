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

// Package rhai contains RHAI (Red Hat AI) specific extensions for the Trainer operator,
// including progression tracking and custom metrics for midstream-specific features.
package rhai

import (
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/kubeflow/trainer/v2/pkg/controller"
	rhaicontroller "github.com/kubeflow/trainer/v2/pkg/rhai/controller"
)

// SetupWithManager wraps the base TrainJob reconciler with RHAI progression tracking.
// Uses decorator pattern to add metrics polling without modifying core functionality.
func SetupWithManager(mgr ctrl.Manager, baseReconciler *controller.TrainJobReconciler) error {
	progressionReconciler := rhaicontroller.NewProgressionReconciler(
		baseReconciler,
		mgr.GetClient(),
	)
	return progressionReconciler.SetupWithManager(mgr)
}
