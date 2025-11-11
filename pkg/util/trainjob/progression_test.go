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

package trainjob

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
)

func TestIsProgressionTrackingEnabled(t *testing.T) {
	tests := []struct {
		name        string
		trainJob    *trainer.TrainJob
		wantEnabled bool
	}{
		{
			name: "enabled",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
			},
			wantEnabled: true,
		},
		{
			name: "disabled",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "disabled",
					},
				},
			},
			wantEnabled: false,
		},
		{
			name: "no annotation",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{},
			},
			wantEnabled: false,
		},
		{
			name:        "nil annotations",
			trainJob:    &trainer.TrainJob{},
			wantEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsProgressionTrackingEnabled(tt.trainJob)
			if got != tt.wantEnabled {
				t.Errorf("IsProgressionTrackingEnabled() = %v, want %v", got, tt.wantEnabled)
			}
		})
	}
}

func TestGetMetricsPort(t *testing.T) {
	tests := []struct {
		name     string
		trainJob *trainer.TrainJob
		wantPort string
	}{
		{
			name: "custom port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationMetricsPort: "8080",
					},
				},
			},
			wantPort: "8080",
		},
		{
			name: "default port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{},
			},
			wantPort: constants.DefaultMetricsPort,
		},
		{
			name:     "nil annotations",
			trainJob: &trainer.TrainJob{},
			wantPort: constants.DefaultMetricsPort,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetMetricsPort(tt.trainJob)
			if got != tt.wantPort {
				t.Errorf("GetMetricsPort() = %v, want %v", got, tt.wantPort)
			}
		})
	}
}

func TestPollTrainingProgress(t *testing.T) {
	tests := []struct {
		name           string
		responseBody   string
		responseStatus int
		wantStatus     *TrainerStatus
		wantErr        bool
	}{
		{
			name: "successful poll with complete metrics",
			responseBody: `{
				"status": "training",
				"status_message": "Training in progress: 45.3% complete",
				"status_details": {
					"last_event": "checkpoint_saved",
					"last_event_time": "2025-11-10 14:30:00",
					"warnings": []
				},
				"progress": {
					"step_current": 4530,
					"step_total": 10000,
					"percent": 45.3,
					"epoch": 2
				},
				"time": {
					"elapsed_sec": 8145.23,
					"elapsed": "2h 15m 45s",
					"remaining_sec": 9854.77,
					"remaining": "2h 44m 14s",
					"started_sec": 1699632000.0,
					"started_at": "2025-11-10 12:00:00",
					"updated_sec": 1699640145.23,
					"updated_at": "2025-11-10 14:15:45"
				},
				"metrics": {
					"loss": 0.234,
					"learning_rate": 0.00005,
					"throughput_samples_sec": 128.5,
					"accuracy": 0.89,
					"f1_score": 0.87
				},
				"checkpoint": {
					"resumed_from": "/output/checkpoint-1000",
					"last_step": 4500,
					"last_path": "/output/checkpoint-4500",
					"saved_at": "2025-11-10 14:30:00"
				}
			}`,
			responseStatus: http.StatusOK,
			wantStatus: &TrainerStatus{
				Status:        "training",
				StatusMessage: "Training in progress: 45.3% complete",
				StatusDetails: StatusDetails{
					LastEvent:     "checkpoint_saved",
					LastEventTime: "2025-11-10 14:30:00",
					Warnings:      []Warning{},
				},
				Progress: Progress{
					StepCurrent: 4530,
					StepTotal:   10000,
					Percent:     45.3,
					Epoch:       2,
				},
				Time: TimeMetrics{
					ElapsedSec:   8145.23,
					Elapsed:      "2h 15m 45s",
					RemainingSec: ptrFloat64(9854.77),
					Remaining:    ptrString("2h 44m 14s"),
					StartedSec:   1699632000.0,
					StartedAt:    "2025-11-10 12:00:00",
					UpdatedSec:   1699640145.23,
					UpdatedAt:    "2025-11-10 14:15:45",
				},
				Metrics: Metrics{
					Loss:                 ptrFloat64(0.234),
					LearningRate:         ptrFloat64(0.00005),
					ThroughputSamplesSec: ptrFloat64(128.5),
					Custom: map[string]interface{}{
						"accuracy": 0.89,
						"f1_score": 0.87,
					},
				},
				Checkpoint: Checkpoint{
					ResumedFrom: "/output/checkpoint-1000",
					LastStep:    4500,
					LastPath:    "/output/checkpoint-4500",
					SavedAt:     "2025-11-10 14:30:00",
				},
			},
			wantErr: false,
		},
		{
			name:           "http error",
			responseBody:   "",
			responseStatus: http.StatusInternalServerError,
			wantStatus:     nil,
			wantErr:        true,
		},
		{
			name:           "invalid json",
			responseBody:   `{invalid json}`,
			responseStatus: http.StatusOK,
			wantStatus:     nil,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.responseStatus)
				w.Write([]byte(tt.responseBody))
			}))
			defer server.Close()

			// Parse server URL to extract host and port
			serverURL := server.URL[7:] // Remove "http://"
			// Split by colon to get host:port
			lastColon := len(serverURL) - 1
			for i := len(serverURL) - 1; i >= 0; i-- {
				if serverURL[i] == ':' {
					lastColon = i
					break
				}
			}
			host := serverURL[:lastColon]
			port := serverURL[lastColon+1:]

			// Create pod with server URL
			pod := &corev1.Pod{
				Status: corev1.PodStatus{
					PodIP: host,
				},
			}

			ctx := context.Background()
			got, err := PollTrainingProgress(ctx, pod, port)

			if (err != nil) != tt.wantErr {
				t.Errorf("PollTrainingProgress() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && got != nil {
				// Compare status
				if diff := cmp.Diff(tt.wantStatus, got); diff != "" {
					t.Errorf("PollTrainingProgress() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestUpdateTrainerStatusAnnotation(t *testing.T) {
	tests := []struct {
		name      string
		trainJob  *trainer.TrainJob
		status    *TrainerStatus
		wantErr   bool
		checkJSON bool
	}{
		{
			name:     "successful update",
			trainJob: &trainer.TrainJob{},
			status: &TrainerStatus{
				Status:        "training",
				StatusMessage: "Training in progress",
				Progress: Progress{
					StepCurrent: 100,
					StepTotal:   1000,
					Percent:     10.0,
					Epoch:       1,
				},
			},
			wantErr:   false,
			checkJSON: true,
		},
		{
			name: "update with custom metrics",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			status: &TrainerStatus{
				Status: "training",
				Metrics: Metrics{
					Loss:         ptrFloat64(0.5),
					LearningRate: ptrFloat64(0.001),
					Custom: map[string]interface{}{
						"accuracy": 0.95,
						"f1_score": 0.92,
					},
				},
			},
			wantErr:   false,
			checkJSON: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := UpdateTrainerStatusAnnotation(tt.trainJob, tt.status)

			if (err != nil) != tt.wantErr {
				t.Errorf("UpdateTrainerStatusAnnotation() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.checkJSON {
				// Verify annotation exists
				if tt.trainJob.Annotations == nil {
					t.Error("Annotations map is nil")
					return
				}

				statusJSON, exists := tt.trainJob.Annotations[constants.AnnotationTrainerStatus]
				if !exists {
					t.Error("Status annotation not found")
					return
				}

				// Verify it's valid JSON
				var decoded map[string]interface{}
				if err := json.Unmarshal([]byte(statusJSON), &decoded); err != nil {
					t.Errorf("Failed to unmarshal status JSON: %v", err)
					return
				}

				// Verify custom metrics are included
				if len(tt.status.Metrics.Custom) > 0 {
					metricsMap, ok := decoded["metrics"].(map[string]interface{})
					if !ok {
						t.Error("Metrics not found in decoded JSON")
						return
					}

					for key := range tt.status.Metrics.Custom {
						if _, exists := metricsMap[key]; !exists {
							t.Errorf("Custom metric %s not found in JSON", key)
						}
					}
				}
			}
		})
	}
}

func TestGetPrimaryPod(t *testing.T) {
	tests := []struct {
		name     string
		trainJob *trainer.TrainJob
		pods     []corev1.Pod
		wantErr  bool
	}{
		{
			name: "running pod found",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
			},
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-job-worker-0",
						Namespace: "default",
						Labels: map[string]string{
							"training.kubeflow.org/job-name": "test-job",
						},
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						PodIP: "10.0.0.1",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "no running pod",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
			},
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-job-worker-0",
						Namespace: "default",
						Labels: map[string]string{
							"training.kubeflow.org/job-name": "test-job",
						},
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodPending,
					},
				},
			},
			wantErr: true,
		},
		{
			name: "no pods",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
			},
			pods:    []corev1.Pod{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = corev1.AddToScheme(scheme)

			objs := make([]client.Object, len(tt.pods))
			for i := range tt.pods {
				objs[i] = &tt.pods[i]
			}

			c := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(objs...).
				Build()

			ctx := context.Background()
			pod, err := GetPrimaryPod(ctx, c, tt.trainJob)

			if (err != nil) != tt.wantErr {
				t.Errorf("GetPrimaryPod() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && pod == nil {
				t.Error("GetPrimaryPod() returned nil pod")
			}
		})
	}
}

// Helper functions
func ptrFloat64(f float64) *float64 {
	return &f
}

func ptrString(s string) *string {
	return &s
}
