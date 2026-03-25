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
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	idxer "github.com/kubeflow/trainer/v2/pkg/runtime/indexer"
	utiltesting "github.com/kubeflow/trainer/v2/pkg/util/testing"
)

func TestReconcile_TrainingRuntimeReconciler(t *testing.T) {
	errorFailedGetTrainingRuntime := errors.New("TEST: failed to get TrainingRuntime")
	cases := map[string]struct {
		trainJobs           trainer.TrainJobList
		trainingRuntime     *trainer.TrainingRuntime
		wantTrainingRuntime *trainer.TrainingRuntime
		wantError           error
	}{
		"no action when runtime with finalizer does not being deleted": {
			trainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Finalizers(constants.ResourceInUseFinalizer).
				Obj(),
			wantTrainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Finalizers(constants.ResourceInUseFinalizer).
				Obj(),
		},
		"remove trainingRuntime due to removed finalizers when runtime without finalizer is deleting": {
			trainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Finalizers(constants.ResourceInUseFinalizer).
				DeletionTimestamp(metav1.Now()).
				Obj(),
			wantError:           errorFailedGetTrainingRuntime,
			wantTrainingRuntime: &trainer.TrainingRuntime{},
		},
		"add finalizer when trainingRuntime is used by trainJob": {
			trainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Obj(),
			trainJobs: trainer.TrainJobList{
				Items: []trainer.TrainJob{
					*utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "trainJob").
						RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "runtime").
						Obj(),
				},
			},
			wantTrainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Finalizers(constants.ResourceInUseFinalizer).
				Obj(),
		},
		"no action when all TrainJobs use another TrainingRuntime": {
			trainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Obj(),
			trainJobs: trainer.TrainJobList{
				Items: []trainer.TrainJob{
					*utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "trainJob").
						RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "another").
						Obj(),
				},
			},
			wantTrainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Obj(),
		},
		"no action when runtime without finalizer is not used by any TrainJob": {
			trainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Obj(),
			wantTrainingRuntime: utiltesting.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "runtime").
				Obj(),
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var cancel func()
			ctx, cancel = context.WithCancel(ctx)
			t.Cleanup(cancel)
			cli := utiltesting.NewClientBuilder().
				WithObjects(tc.trainingRuntime).
				WithIndex(&trainer.TrainJob{}, idxer.TrainJobRuntimeRefKey, idxer.IndexTrainJobTrainingRuntime).
				WithLists(&tc.trainJobs).
				WithInterceptorFuncs(interceptor.Funcs{
					Get: func(ctx context.Context, cli client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
						if _, ok := obj.(*trainer.TrainJob); !ok && errors.Is(tc.wantError, errorFailedGetTrainingRuntime) {
							return errorFailedGetTrainingRuntime
						}
						return cli.Get(ctx, key, obj, opts...)
					},
				}).
				Build()
			r := NewTrainingRuntimeReconciler(cli, nil)
			runtimeKey := client.ObjectKeyFromObject(tc.trainingRuntime)
			_, gotError := r.Reconcile(ctx, reconcile.Request{NamespacedName: runtimeKey})
			if diff := cmp.Diff(tc.wantError, gotError, cmpopts.EquateErrors()); len(diff) != 0 {
				t.Errorf("Unexpected Recincile error: (-want, +got): \n%s", diff)
			}
			var gotRuntime trainer.TrainingRuntime
			gotError = cli.Get(ctx, runtimeKey, &gotRuntime)
			if diff := cmp.Diff(tc.wantError, gotError, cmpopts.EquateErrors()); len(diff) != 0 {
				t.Errorf("Unexpected GET error: (-want, +got): \n%s", diff)
			}
			if diff := cmp.Diff(tc.wantTrainingRuntime, &gotRuntime,
				cmpopts.IgnoreFields(metav1.ObjectMeta{}, "ResourceVersion"),
				cmpopts.IgnoreFields(metav1.TypeMeta{}, "Kind", "APIVersion"),
			); len(diff) != 0 {
				t.Errorf("Unexpected TrainingRuntime: (-want, +got): \n%s", diff)
			}
		})
	}
}

func TestNotifyTrainJobUpdate_TrainingRuntimeReconciler(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		oldJob    *trainer.TrainJob
		newJob    *trainer.TrainJob
		wantEvent event.TypedGenericEvent[iter.Seq[types.NamespacedName]]
	}{
		"UPDATE Event: runtimeRef is TrainingRuntime": {
			oldJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "test-runtime").
				Obj(),
			newJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "test-runtime").
				SpecLabel("key", "value").
				Obj(),
			wantEvent: event.TypedGenericEvent[iter.Seq[types.NamespacedName]]{
				Object: func(yield func(types.NamespacedName) bool) {
					yield(types.NamespacedName{Namespace: metav1.NamespaceDefault, Name: "test-runtime"})
				},
			},
		},
		"UPDATE Event: runtimeRef is not TrainingRuntime": {
			oldJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				Obj(),
			newJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				SpecLabel("key", "value").
				Obj(),
		},
		"CREATE Event: runtimeRef is TrainingRuntime": {
			newJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "test-runtime").
				Obj(),
			wantEvent: event.TypedGenericEvent[iter.Seq[types.NamespacedName]]{
				Object: func(yield func(types.NamespacedName) bool) {
					yield(types.NamespacedName{Namespace: metav1.NamespaceDefault, Name: "test-runtime"})
				},
			},
		},
		"CREATE Event: runtimeRef is not TrainingRuntime": {
			newJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				Obj(),
		},
		"DELETE Event: runtimeRef is TrainingRuntime": {
			oldJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), "test-runtime").
				Obj(),
			wantEvent: event.TypedGenericEvent[iter.Seq[types.NamespacedName]]{
				Object: func(yield func(types.NamespacedName) bool) {
					yield(types.NamespacedName{Namespace: metav1.NamespaceDefault, Name: "test-runtime"})
				},
			},
		},
		"DELETE Event: runtimeRef is not TrainingRuntime": {
			oldJob: utiltesting.MakeTrainJobWrapper(metav1.NamespaceDefault, "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				Obj(),
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			logger, _ := ktesting.NewTestContext(t)
			updateCh := make(chan event.TypedGenericEvent[iter.Seq[types.NamespacedName]], 1)
			t.Cleanup(func() {
				close(updateCh)
			})
			r := &TrainingRuntimeReconciler{
				log:                      logger,
				nonRuntimeObjectUpdateCh: updateCh,
			}
			r.NotifyTrainJobUpdate(tc.oldJob, tc.newJob)
			var got event.TypedGenericEvent[iter.Seq[types.NamespacedName]]
			select {
			case got = <-updateCh:
			case <-time.After(time.Second):
			}
			if diff := cmp.Diff(tc.wantEvent, got, utiltesting.TrainJobUpdateReconcileRequestCmpOpts); len(diff) != 0 {
				t.Errorf("Unexpected GenericEvent (-want, +got):\n%s", diff)
			}
		})
	}
}

// TestGracefulShutdown_TrainingRuntimeReconciler tests the graceful shutdown behavior
// for RHOAIENG-42316: Stuck Trainer CR when disabling Trainer component
func TestGracefulShutdown_TrainingRuntimeReconciler(t *testing.T) {
	cases := map[string]struct {
		runtimes                []trainer.TrainingRuntime
		wantRemainingFinalizers map[string][]string // runtime namespace/name -> finalizers
		wantDeleted             []string            // runtime namespace/name that should be deleted
	}{
		"removes finalizer from terminating TrainingRuntime": {
			runtimes: []trainer.TrainingRuntime{
				*utiltesting.MakeTrainingRuntimeWrapper("default", "runtime-terminating").
					Finalizers(constants.ResourceInUseFinalizer).
					DeletionTimestamp(metav1.Now()).
					Obj(),
			},
			wantDeleted: []string{"default/runtime-terminating"}, // Deleted after finalizer removed
		},
		"keeps finalizer on non-terminating TrainingRuntime": {
			runtimes: []trainer.TrainingRuntime{
				*utiltesting.MakeTrainingRuntimeWrapper("default", "runtime-active").
					Finalizers(constants.ResourceInUseFinalizer).
					Obj(),
			},
			wantRemainingFinalizers: map[string][]string{
				"default/runtime-active": {constants.ResourceInUseFinalizer}, // Finalizer should remain
			},
		},
		"handles multiple runtimes across namespaces": {
			runtimes: []trainer.TrainingRuntime{
				*utiltesting.MakeTrainingRuntimeWrapper("ns1", "runtime-terminating").
					Finalizers(constants.ResourceInUseFinalizer).
					DeletionTimestamp(metav1.Now()).
					Obj(),
				*utiltesting.MakeTrainingRuntimeWrapper("ns2", "runtime-active").
					Finalizers(constants.ResourceInUseFinalizer).
					Obj(),
				*utiltesting.MakeTrainingRuntimeWrapper("ns3", "runtime-terminating-multi").
					Finalizers(constants.ResourceInUseFinalizer, "another-finalizer").
					DeletionTimestamp(metav1.Now()).
					Obj(),
			},
			wantDeleted: []string{"ns1/runtime-terminating"}, // Deleted after last finalizer removed
			wantRemainingFinalizers: map[string][]string{
				"ns2/runtime-active":            {constants.ResourceInUseFinalizer}, // Kept (not terminating)
				"ns3/runtime-terminating-multi": {"another-finalizer"},              // Only ResourceInUseFinalizer removed
			},
		},
		"does not fail on runtime without finalizer": {
			runtimes: []trainer.TrainingRuntime{
				*utiltesting.MakeTrainingRuntimeWrapper("default", "runtime-active-no-finalizer").
					Obj(),
			},
			wantRemainingFinalizers: map[string][]string{
				"default/runtime-active-no-finalizer": nil, // No finalizer to remove, not terminating
			},
		},
		"handles empty list": {
			runtimes:                []trainer.TrainingRuntime{},
			wantRemainingFinalizers: map[string][]string{},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()

			// Build client with test resources
			builder := utiltesting.NewClientBuilder()
			for i := range tc.runtimes {
				builder = builder.WithObjects(&tc.runtimes[i])
			}
			cli := builder.Build()

			r := NewTrainingRuntimeReconciler(cli, nil)

			// Call GracefulShutdown
			err := r.GracefulShutdown(ctx)
			if err != nil {
				t.Errorf("GracefulShutdown returned unexpected error: %v", err)
			}

			// Helper to parse namespace/name from key
			parseKey := func(runtimeKey string) (string, string) {
				for i, ch := range runtimeKey {
					if ch == '/' {
						return runtimeKey[:i], runtimeKey[i+1:]
					}
				}
				return "", ""
			}

			// Verify resources that should be deleted
			for _, runtimeKey := range tc.wantDeleted {
				namespace, name := parseKey(runtimeKey)
				var gotRuntime trainer.TrainingRuntime
				err := cli.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, &gotRuntime)
				if err == nil {
					t.Errorf("Expected TrainingRuntime %s to be deleted, but it still exists", runtimeKey)
				}
			}

			// Verify finalizers on resources that should remain
			for runtimeKey, wantFinalizers := range tc.wantRemainingFinalizers {
				namespace, name := parseKey(runtimeKey)
				var gotRuntime trainer.TrainingRuntime
				err := cli.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, &gotRuntime)
				if err != nil {
					t.Errorf("Failed to get TrainingRuntime %s: %v", runtimeKey, err)
					continue
				}

				if diff := cmp.Diff(wantFinalizers, gotRuntime.Finalizers); len(diff) != 0 {
					t.Errorf("Unexpected finalizers for %s (-want, +got):\n%s", runtimeKey, diff)
				}
			}
		})
	}
}
