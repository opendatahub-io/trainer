/*
Copyright 2025 The Kubeflow Authors.

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
	"iter"
	"slices"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	ctrlutil "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	idxer "github.com/kubeflow/trainer/v2/pkg/runtime/indexer"
	"github.com/kubeflow/trainer/v2/pkg/util/trainjob"
)

const (
	updateChBuffer = 10
)

type TrainingRuntimeReconciler struct {
	log                      logr.Logger
	client                   client.Client
	recorder                 record.EventRecorder
	nonRuntimeObjectUpdateCh chan event.TypedGenericEvent[iter.Seq[types.NamespacedName]]
}

var _ reconcile.Reconciler = (*TrainingRuntimeReconciler)(nil)
var _ TrainJobWatcher = (*TrainingRuntimeReconciler)(nil)

func NewTrainingRuntimeReconciler(cli client.Client, recorder record.EventRecorder) *TrainingRuntimeReconciler {
	return &TrainingRuntimeReconciler{
		log:                      ctrl.Log.WithName("trainingruntime-controller"),
		client:                   cli,
		recorder:                 recorder,
		nonRuntimeObjectUpdateCh: make(chan event.TypedGenericEvent[iter.Seq[types.NamespacedName]], updateChBuffer),
	}
}

// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=trainingruntimes,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=trainingruntimes/finalizers,verbs=get;update;patch

func (r *TrainingRuntimeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var runtime trainer.TrainingRuntime
	if err := r.client.Get(ctx, req.NamespacedName, &runtime); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	log := ctrl.LoggerFrom(ctx).WithValues("trainingRuntime", klog.KObj(&runtime))
	ctrl.LoggerInto(ctx, log)
	log.V(2).Info("Reconciling TrainingRuntime")

	trainJobs := &trainer.TrainJobList{}
	if err := r.client.List(ctx, trainJobs, client.InNamespace(runtime.Namespace), client.MatchingFields{
		idxer.TrainJobRuntimeRefKey: runtime.Name,
	}); err != nil {
		return ctrl.Result{}, err
	}
	if !ctrlutil.ContainsFinalizer(&runtime, constants.ResourceInUseFinalizer) && len(trainJobs.Items) != 0 {
		ctrlutil.AddFinalizer(&runtime, constants.ResourceInUseFinalizer)
		return ctrl.Result{}, r.client.Update(ctx, &runtime)
	} else if !runtime.DeletionTimestamp.IsZero() && len(trainJobs.Items) == 0 {
		ctrlutil.RemoveFinalizer(&runtime, constants.ResourceInUseFinalizer)
		return ctrl.Result{}, r.client.Update(ctx, &runtime)
	}
	return ctrl.Result{}, nil
}

// GracefulShutdown removes finalizers from TrainingRuntimes during operator shutdown.
// This prevents resources from being stuck in Terminating state when the operator is terminated
// before it can complete normal finalizer cleanup.
// Fix for RHOAIENG-42316: Stuck Trainer CR when disabling Trainer component.
func (r *TrainingRuntimeReconciler) GracefulShutdown(ctx context.Context) error {
	log := r.log.WithName("graceful-shutdown")
	log.Info("Starting graceful shutdown for TrainingRuntimes")

	// List all TrainingRuntimes across all namespaces
	var runtimeList trainer.TrainingRuntimeList
	if err := r.client.List(ctx, &runtimeList); err != nil {
		log.Error(err, "Failed to list TrainingRuntimes during graceful shutdown")
		return err
	}

	// Process TrainingRuntimes that are being deleted
	for i := range runtimeList.Items {
		runtime := &runtimeList.Items[i]

		// Skip if not being deleted
		if runtime.DeletionTimestamp.IsZero() {
			continue
		}

		// Skip if finalizer not present
		if !ctrlutil.ContainsFinalizer(runtime, constants.ResourceInUseFinalizer) {
			continue
		}

		log.Info("Removing finalizer from terminating TrainingRuntime",
			"namespace", runtime.Namespace, "name", runtime.Name)

		// Remove the finalizer to allow deletion
		ctrlutil.RemoveFinalizer(runtime, constants.ResourceInUseFinalizer)
		if err := r.client.Update(ctx, runtime); err != nil {
			log.Error(err, "Failed to remove finalizer during graceful shutdown",
				"namespace", runtime.Namespace, "name", runtime.Name)
			// Continue processing others even if one fails
			continue
		}

		log.Info("Successfully removed finalizer during graceful shutdown",
			"namespace", runtime.Namespace, "name", runtime.Name)
	}

	log.Info("Graceful shutdown completed for TrainingRuntimes")
	return nil
}

func (r *TrainingRuntimeReconciler) NotifyTrainJobUpdate(oldJob, newJob *trainer.TrainJob) {
	var runtimeNSName *types.NamespacedName
	switch {
	case oldJob != nil && newJob != nil:
		// UPDATE Event.
		if trainjob.RuntimeRefIsTrainingRuntime(newJob.Spec.RuntimeRef) {
			runtimeNSName = &types.NamespacedName{Namespace: newJob.Namespace, Name: newJob.Spec.RuntimeRef.Name}
		}
	case oldJob == nil:
		// CREATE Event.
		if trainjob.RuntimeRefIsTrainingRuntime(newJob.Spec.RuntimeRef) {
			runtimeNSName = &types.NamespacedName{Namespace: newJob.Namespace, Name: newJob.Spec.RuntimeRef.Name}
		}
	default:
		// DELETE Event.
		if trainjob.RuntimeRefIsTrainingRuntime(oldJob.Spec.RuntimeRef) {
			runtimeNSName = &types.NamespacedName{Namespace: oldJob.Namespace, Name: oldJob.Spec.RuntimeRef.Name}
		}
	}
	if runtimeNSName != nil {
		r.nonRuntimeObjectUpdateCh <- event.TypedGenericEvent[iter.Seq[types.NamespacedName]]{
			Object: slices.Values([]types.NamespacedName{*runtimeNSName}),
		}
	}
}

type nonRuntimeObjectHandler struct{}

var _ handler.TypedEventHandler[iter.Seq[types.NamespacedName], reconcile.Request] = (*nonRuntimeObjectHandler)(nil)

func (h *nonRuntimeObjectHandler) Create(context.Context, event.TypedCreateEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonRuntimeObjectHandler) Update(context.Context, event.TypedUpdateEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonRuntimeObjectHandler) Delete(context.Context, event.TypedDeleteEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonRuntimeObjectHandler) Generic(_ context.Context, e event.TypedGenericEvent[iter.Seq[types.NamespacedName]], q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
	for nsName := range e.Object {
		q.AddAfter(reconcile.Request{NamespacedName: nsName}, time.Second)
	}
}

func (r *TrainingRuntimeReconciler) SetupWithManager(mgr ctrl.Manager, options controller.Options) error {
	return builder.TypedControllerManagedBy[reconcile.Request](mgr).
		Named("trainingruntime_controller").
		WithOptions(options).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&trainer.TrainingRuntime{},
			&handler.TypedEnqueueRequestForObject[*trainer.TrainingRuntime]{},
		)).
		WatchesRawSource(source.Channel(r.nonRuntimeObjectUpdateCh, &nonRuntimeObjectHandler{})).
		Complete(r)
}
