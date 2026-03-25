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
	"time"

	"github.com/go-logr/logr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/runtime"
)

// Reconcilers holds reconciler instances for graceful shutdown handling.
type Reconcilers struct {
	TrainingRuntime        *TrainingRuntimeReconciler
	ClusterTrainingRuntime *ClusterTrainingRuntimeReconciler
}

// GracefulShutdownRunnable implements the controller-runtime Runnable interface
// to perform graceful shutdown when the manager's context is cancelled.
type GracefulShutdownRunnable struct {
	ClusterTrainingRuntime *ClusterTrainingRuntimeReconciler
	TrainingRuntime        *TrainingRuntimeReconciler
	log                    logr.Logger
}

// Start blocks until the context is cancelled (SIGTERM received), then performs graceful shutdown.
// This is called by the controller-runtime manager and provides a window where the webhook
// server is still draining but the context is cancelled, allowing client.Update() to succeed.
func (r *GracefulShutdownRunnable) Start(ctx context.Context) error {
	// Block until context is cancelled (SIGTERM received)
	<-ctx.Done()

	r.log.Info("Performing graceful shutdown")

	// Create new context with timeout for cleanup operations
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if r.ClusterTrainingRuntime != nil {
		if err := r.ClusterTrainingRuntime.GracefulShutdown(shutdownCtx); err != nil {
			r.log.Error(err, "Error during ClusterTrainingRuntime graceful shutdown")
		}
	}

	if r.TrainingRuntime != nil {
		if err := r.TrainingRuntime.GracefulShutdown(shutdownCtx); err != nil {
			r.log.Error(err, "Error during TrainingRuntime graceful shutdown")
		}
	}

	r.log.Info("Graceful shutdown completed")
	return nil
}

func SetupControllers(mgr ctrl.Manager, runtimes map[string]runtime.Runtime, options controller.Options) (string, *Reconcilers, error) {
	runtimeRec := NewTrainingRuntimeReconciler(
		mgr.GetClient(),
		mgr.GetEventRecorderFor("trainer-trainingruntime-controller"),
	)
	if err := runtimeRec.SetupWithManager(mgr, options); err != nil {
		return trainer.TrainingRuntimeKind, nil, err
	}
	clRuntimeRec := NewClusterTrainingRuntimeReconciler(
		mgr.GetClient(),
		mgr.GetEventRecorderFor("trainer-clustertrainingruntime-controller"),
	)
	if err := clRuntimeRec.SetupWithManager(mgr, options); err != nil {
		return trainer.ClusterTrainingRuntimeKind, nil, err
	}
	if err := NewTrainJobReconciler(
		mgr.GetClient(),
		mgr.GetAPIReader(),
		mgr.GetEventRecorderFor("trainer-trainjob-controller"),
		runtimes,
		WithWatchers(runtimeRec, clRuntimeRec),
	).SetupWithManager(mgr, options); err != nil {
		return trainer.TrainJobKind, nil, err
	}

	// Register the graceful shutdown runnable
	// This will be called when the manager's context is cancelled (SIGTERM),
	// providing a window where the webhook server is still draining but client.Update() can succeed
	shutdownRunnable := &GracefulShutdownRunnable{
		ClusterTrainingRuntime: clRuntimeRec,
		TrainingRuntime:        runtimeRec,
		log:                    ctrl.Log.WithName("graceful-shutdown"),
	}
	if err := mgr.Add(shutdownRunnable); err != nil {
		return "graceful-shutdown-runnable", nil, err
	}

	// Return reconciler instances for backward compatibility
	reconcilers := &Reconcilers{
		TrainingRuntime:        runtimeRec,
		ClusterTrainingRuntime: clRuntimeRec,
	}
	return "", reconcilers, nil
}
