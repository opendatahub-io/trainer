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

package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/api/meta"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/controller"
	"github.com/kubeflow/trainer/v2/pkg/rhai/progression"
)

const (
	logLevelInfo  = 1 // Important events (errors, completion)
	logLevelDebug = 2 // Debug info (requeue intervals)
)

// ProgressionReconciler wraps TrainJobReconciler to add RHAI progression tracking.
// Polls metrics from pods and updates TrainJob annotations when tracking is enabled.
type ProgressionReconciler struct {
	base   *controller.TrainJobReconciler
	client client.Client
}

// NewProgressionReconciler creates a new ProgressionReconciler.
func NewProgressionReconciler(base *controller.TrainJobReconciler, client client.Client) *ProgressionReconciler {
	return &ProgressionReconciler{
		base:   base,
		client: client,
	}
}

// Reconcile wraps the base reconciler and adds progression tracking logic.
func (r *ProgressionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Call base reconciler first
	result, err := r.base.Reconcile(ctx, req)

	// Get the TrainJob to check progression tracking
	var trainJob trainer.TrainJob
	if getErr := r.client.Get(ctx, req.NamespacedName, &trainJob); getErr != nil {
		// If we can't get the TrainJob, just return base result
		return result, err
	}

	log := ctrl.LoggerFrom(ctx)

	// RHAI-specific: Poll training progress for running jobs with progression tracking enabled
	if !progression.IsProgressionTrackingEnabled(&trainJob) {
		return result, err
	}

	isRunning := !meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobSuspended) &&
		!meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobComplete) &&
		!meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobFailed)

	isCompleted := meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobComplete)
	isFailed := meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobFailed)

	if isRunning {
		// Poll metrics while job is running
		if _, pollErr := progression.PollAndUpdateProgress(ctx, r.client, &trainJob); pollErr != nil {
			log.V(logLevelInfo).Info("Failed to poll training progress", "error", pollErr)
		} else {
			log.V(logLevelDebug).Info("Successfully updated training progress")
		}
		// Requeue to continue polling while job is running
		pollInterval := progression.GetMetricsPollInterval(&trainJob)
		log.V(logLevelDebug).Info("Requeuing for metrics polling", "interval", pollInterval)
		return ctrl.Result{RequeueAfter: pollInterval}, err
	}

	if (isCompleted || isFailed) && !progression.IsFinalStatusCaptured(&trainJob) {
		// Job just completed/failed - do one final metrics poll to capture completion status
		if _, pollErr := progression.PollAndUpdateFinalProgress(ctx, r.client, &trainJob, isCompleted); pollErr != nil {
			log.V(logLevelInfo).Info("Failed to capture final training progress", "error", pollErr, "completed", isCompleted)
		} else {
			log.Info("Captured final training progress", "completed", isCompleted)
		}
	}

	return result, err
}

// SetupWithManager sets up the controller with the Manager.
func (r *ProgressionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainer.TrainJob{}).
		Named("trainjob-progression-controller").
		Complete(r)
}
