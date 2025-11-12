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
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/controller"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
	"github.com/kubeflow/trainer/v2/pkg/rhai/progression"
)

func TestNewProgressionReconciler(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = trainer.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	baseReconciler := &controller.TrainJobReconciler{}

	reconciler := NewProgressionReconciler(baseReconciler, fakeClient)

	if reconciler == nil {
		t.Fatal("Expected reconciler to be created, got nil")
	}

	if reconciler.client != fakeClient {
		t.Error("Expected reconciler client to match provided client")
	}

	if reconciler.base != baseReconciler {
		t.Error("Expected reconciler base to match provided base reconciler")
	}
}

func TestProgressionReconciler_DeterminesRequeueLogic(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = trainer.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	tests := []struct {
		name              string
		trainJob          *trainer.TrainJob
		expectRequeue     bool
		expectInterval    bool
		expectDescription string
	}{
		{
			name: "Job without progression tracking - no requeue",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobSuspended,
							Status: metav1.ConditionFalse,
						},
					},
				},
			},
			expectRequeue:     false,
			expectInterval:    false,
			expectDescription: "Progression tracking disabled, should not requeue",
		},
		{
			name: "Job with progression tracking and running - should requeue",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobSuspended,
							Status: metav1.ConditionFalse,
						},
						{
							Type:   trainer.TrainJobComplete,
							Status: metav1.ConditionFalse,
						},
						{
							Type:   trainer.TrainJobFailed,
							Status: metav1.ConditionFalse,
						},
					},
				},
			},
			expectRequeue:     true,
			expectInterval:    true,
			expectDescription: "Running job with progression tracking should requeue for polling",
		},
		{
			name: "Job with progression tracking but suspended - no requeue",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobSuspended,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectRequeue:     false,
			expectInterval:    false,
			expectDescription: "Suspended job should not requeue even with progression tracking",
		},
		{
			name: "Job with progression tracking and completed - no requeue",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobComplete,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectRequeue:     false,
			expectInterval:    false,
			expectDescription: "Completed job should not requeue",
		},
		{
			name: "Job with progression tracking and failed - no requeue",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobFailed,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectRequeue:     false,
			expectInterval:    false,
			expectDescription: "Failed job should not requeue",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake client with the TrainJob
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(tt.trainJob).
				WithStatusSubresource(&trainer.TrainJob{}).
				Build()

			// Test the decision logic without actually calling Reconcile
			// This tests the core logic of whether progression tracking should requeue
			isEnabled := progression.IsProgressionTrackingEnabled(tt.trainJob)

			isRunning := true
			for _, cond := range tt.trainJob.Status.Conditions {
				if cond.Type == trainer.TrainJobSuspended && cond.Status == metav1.ConditionTrue {
					isRunning = false
					break
				}
				if cond.Type == trainer.TrainJobComplete && cond.Status == metav1.ConditionTrue {
					isRunning = false
					break
				}
				if cond.Type == trainer.TrainJobFailed && cond.Status == metav1.ConditionTrue {
					isRunning = false
					break
				}
			}

			shouldRequeue := isEnabled && isRunning
			shouldHaveInterval := shouldRequeue

			if shouldRequeue != tt.expectRequeue {
				t.Errorf("%s: expected requeue=%v, got %v", tt.expectDescription, tt.expectRequeue, shouldRequeue)
			}

			if shouldHaveInterval != tt.expectInterval {
				t.Errorf("%s: expected interval=%v, got %v", tt.expectDescription, tt.expectInterval, shouldHaveInterval)
			}

			// Verify the client has the job
			var retrievedJob trainer.TrainJob
			err := fakeClient.Get(context.Background(), types.NamespacedName{
				Name:      tt.trainJob.Name,
				Namespace: tt.trainJob.Namespace,
			}, &retrievedJob)
			if err != nil {
				t.Fatalf("Failed to retrieve TrainJob: %v", err)
			}

			if retrievedJob.Name != tt.trainJob.Name {
				t.Errorf("Expected job name %s, got %s", tt.trainJob.Name, retrievedJob.Name)
			}
		})
	}
}

func TestProgressionReconciler_RequeueInterval(t *testing.T) {
	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-job",
			Namespace: "default",
			Annotations: map[string]string{
				constants.AnnotationProgressionTracking: "enabled",
			},
		},
	}

	// Test default interval
	defaultInterval := progression.GetMetricsPollInterval(trainJob)
	expectedDefault := time.Duration(constants.DefaultMetricsPollIntervalSecs) * time.Second
	if defaultInterval != expectedDefault {
		t.Errorf("Expected default interval %v, got %v", expectedDefault, defaultInterval)
	}

	// Test custom interval
	trainJob.Annotations[constants.AnnotationMetricsPollInterval] = "60s"
	customInterval := progression.GetMetricsPollInterval(trainJob)
	expectedCustom := 60 * time.Second
	if customInterval != expectedCustom {
		t.Errorf("Expected custom interval %v, got %v", expectedCustom, customInterval)
	}
}
