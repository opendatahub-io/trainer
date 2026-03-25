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

type ClusterTrainingRuntimeReconciler struct {
	log                        logr.Logger
	client                     client.Client
	recorder                   record.EventRecorder
	nonClRuntimeObjectUpdateCh chan event.TypedGenericEvent[iter.Seq[types.NamespacedName]]
}

var _ reconcile.Reconciler = (*ClusterTrainingRuntimeReconciler)(nil)
var _ TrainJobWatcher = (*ClusterTrainingRuntimeReconciler)(nil)

func NewClusterTrainingRuntimeReconciler(cli client.Client, recorder record.EventRecorder) *ClusterTrainingRuntimeReconciler {
	return &ClusterTrainingRuntimeReconciler{
		log:                        ctrl.Log.WithName("clustertrainingruntime-controller"),
		client:                     cli,
		recorder:                   recorder,
		nonClRuntimeObjectUpdateCh: make(chan event.TypedGenericEvent[iter.Seq[types.NamespacedName]], updateChBuffer),
	}
}

// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=clustertrainingruntimes,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=clustertrainingruntimes/finalizers,verbs=get;update;patch

func (r *ClusterTrainingRuntimeReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	var clRuntime trainer.ClusterTrainingRuntime
	if err := r.client.Get(ctx, request.NamespacedName, &clRuntime); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	log := r.log.WithValues("clusterTrainingRuntime", klog.KObj(&clRuntime))
	ctrl.LoggerInto(ctx, log)
	log.V(2).Info("Reconciling ClusterTrainingRuntime")

	trainJobs := &trainer.TrainJobList{}
	if err := r.client.List(ctx, trainJobs, client.MatchingFields{
		idxer.TrainJobClusterRuntimeRefKey: clRuntime.Name,
	}); err != nil {
		return ctrl.Result{}, err
	}
	if !ctrlutil.ContainsFinalizer(&clRuntime, constants.ResourceInUseFinalizer) && len(trainJobs.Items) != 0 {
		ctrlutil.AddFinalizer(&clRuntime, constants.ResourceInUseFinalizer)
		return ctrl.Result{}, r.client.Update(ctx, &clRuntime)
	} else if !clRuntime.DeletionTimestamp.IsZero() && len(trainJobs.Items) == 0 {
		ctrlutil.RemoveFinalizer(&clRuntime, constants.ResourceInUseFinalizer)
		return ctrl.Result{}, r.client.Update(ctx, &clRuntime)
	}
	return ctrl.Result{}, nil
}

// GracefulShutdown removes finalizers from ClusterTrainingRuntimes during operator shutdown.
// This prevents resources from being stuck in Terminating state when the operator is terminated
// before it can complete normal finalizer cleanup.
// Fix for RHOAIENG-42316: Stuck Trainer CR when disabling Trainer component.
func (r *ClusterTrainingRuntimeReconciler) GracefulShutdown(ctx context.Context) error {
	log := r.log.WithName("graceful-shutdown")
	log.Info("Starting graceful shutdown for ClusterTrainingRuntimes")

	// List all ClusterTrainingRuntimes
	var clRuntimeList trainer.ClusterTrainingRuntimeList
	if err := r.client.List(ctx, &clRuntimeList); err != nil {
		log.Error(err, "Failed to list ClusterTrainingRuntimes during graceful shutdown")
		return err
	}

	// Process ClusterTrainingRuntimes that are being deleted
	for i := range clRuntimeList.Items {
		clRuntime := &clRuntimeList.Items[i]

		// Skip if not being deleted
		if clRuntime.DeletionTimestamp.IsZero() {
			continue
		}

		// Skip if finalizer not present
		if !ctrlutil.ContainsFinalizer(clRuntime, constants.ResourceInUseFinalizer) {
			continue
		}

		log.Info("Removing finalizer from terminating ClusterTrainingRuntime",
			"name", clRuntime.Name)

		// Remove the finalizer using the standard controller-runtime API
		ctrlutil.RemoveFinalizer(clRuntime, constants.ResourceInUseFinalizer)
		if err := r.client.Update(ctx, clRuntime); err != nil {
			log.Error(err, "Failed to remove finalizer during graceful shutdown",
				"name", clRuntime.Name)
			// Continue processing others even if one fails
			continue
		}

		log.Info("Successfully removed finalizer during graceful shutdown",
			"name", clRuntime.Name)
	}

	log.Info("Graceful shutdown completed for ClusterTrainingRuntimes")
	return nil
}

func (r *ClusterTrainingRuntimeReconciler) NotifyTrainJobUpdate(oldJob, newJob *trainer.TrainJob) {
	var clRuntimeNSName *types.NamespacedName
	switch {
	case oldJob != nil && newJob != nil:
		// UPDATE Event.
		if trainjob.RuntimeRefIsClusterTrainingRuntime(newJob.Spec.RuntimeRef) {
			clRuntimeNSName = &types.NamespacedName{Name: newJob.Spec.RuntimeRef.Name}
		}
	case oldJob == nil:
		// CREATE Event.
		if trainjob.RuntimeRefIsClusterTrainingRuntime(newJob.Spec.RuntimeRef) {
			clRuntimeNSName = &types.NamespacedName{Name: newJob.Spec.RuntimeRef.Name}
		}
	default:
		// DELETE Event.
		if trainjob.RuntimeRefIsClusterTrainingRuntime(oldJob.Spec.RuntimeRef) {
			clRuntimeNSName = &types.NamespacedName{Name: oldJob.Spec.RuntimeRef.Name}
		}
	}
	if clRuntimeNSName != nil {
		r.nonClRuntimeObjectUpdateCh <- event.TypedGenericEvent[iter.Seq[types.NamespacedName]]{
			Object: slices.Values([]types.NamespacedName{*clRuntimeNSName}),
		}
	}
}

type nonClRuntimeObjectHandler struct{}

var _ handler.TypedEventHandler[iter.Seq[types.NamespacedName], reconcile.Request] = (*nonClRuntimeObjectHandler)(nil)

func (h *nonClRuntimeObjectHandler) Create(context.Context, event.TypedCreateEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonClRuntimeObjectHandler) Update(context.Context, event.TypedUpdateEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonClRuntimeObjectHandler) Delete(context.Context, event.TypedDeleteEvent[iter.Seq[types.NamespacedName]], workqueue.TypedRateLimitingInterface[reconcile.Request]) {
}
func (h *nonClRuntimeObjectHandler) Generic(_ context.Context, e event.TypedGenericEvent[iter.Seq[types.NamespacedName]], q workqueue.TypedRateLimitingInterface[reconcile.Request]) {
	for nsName := range e.Object {
		q.AddAfter(reconcile.Request{NamespacedName: nsName}, time.Second)
	}
}

func (r *ClusterTrainingRuntimeReconciler) SetupWithManager(mgr ctrl.Manager, options controller.Options) error {
	return builder.TypedControllerManagedBy[reconcile.Request](mgr).
		Named("clustertrainingruntime_controller").
		WithOptions(options).
		WatchesRawSource(source.TypedKind(
			mgr.GetCache(),
			&trainer.ClusterTrainingRuntime{},
			&handler.TypedEnqueueRequestForObject[*trainer.ClusterTrainingRuntime]{},
		)).
		WatchesRawSource(source.Channel(r.nonClRuntimeObjectUpdateCh, &nonClRuntimeObjectHandler{})).
		Complete(r)
}
