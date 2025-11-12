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

package progression

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
)

func TestIsProgressionTrackingEnabled(t *testing.T) {
	tests := []struct {
		name        string
		trainJob    *trainer.TrainJob
		wantEnabled bool
	}{
		{
			name: "enabled with true",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "true",
					},
				},
			},
			wantEnabled: true,
		},
		{
			name: "other values not accepted",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "enabled",
					},
				},
			},
			wantEnabled: false,
		},
		{
			name: "false value",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "false",
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

func TestGetMetricsPollInterval(t *testing.T) {
	tests := []struct {
		name     string
		interval string
		expected time.Duration
	}{
		{
			name:     "duration format seconds",
			interval: "30s",
			expected: 30 * time.Second,
		},
		{
			name:     "duration format minutes",
			interval: "1m",
			expected: 60 * time.Second,
		},
		{
			name:     "integer format",
			interval: "45",
			expected: 45 * time.Second,
		},
		{
			name:     "below minimum clamped to 5s",
			interval: "3s",
			expected: 5 * time.Second,
		},
		{
			name:     "above maximum clamped to 300s",
			interval: "600s",
			expected: 300 * time.Second,
		},
		{
			name:     "invalid format uses default",
			interval: "invalid",
			expected: 30 * time.Second,
		},
		{
			name:     "missing annotation uses default",
			interval: "",
			expected: 30 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trainJob := &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationMetricsPollInterval: tt.interval,
					},
				},
			}

			got := GetMetricsPollInterval(trainJob)
			if got != tt.expected {
				t.Errorf("GetMetricsPollInterval() = %v, want %v", got, tt.expected)
			}
		})
	}

	// Test nil annotations map
	t.Run("nil annotations uses default", func(t *testing.T) {
		trainJob := &trainer.TrainJob{}
		got := GetMetricsPollInterval(trainJob)
		expected := 30 * time.Second
		if got != expected {
			t.Errorf("GetMetricsPollInterval() = %v, want %v", got, expected)
		}
	})
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
				"progressPercentage": 45,
				"estimatedRemainingSeconds": 9855,
				"currentStep": 4530,
				"totalSteps": 10000,
				"currentEpoch": 2,
				"totalEpochs": 5,
				"trainMetrics": {
					"loss": 0.234,
					"learning_rate": 0.00005,
					"throughput_samples_sec": 128.5
				},
				"evalMetrics": {
					"eval_accuracy": 0.89,
					"eval_f1_score": 0.87
				}
			}`,
			responseStatus: http.StatusOK,
			wantStatus: &TrainerStatus{
				ProgressPercentage:        ptrInt(45),
				EstimatedRemainingSeconds: ptrInt(9855),
				CurrentStep:               4530,
				TotalSteps:                ptrInt(10000),
				CurrentEpoch:              2,
				TotalEpochs:               ptrInt(5),
				TrainMetrics: map[string]interface{}{
					"loss":                   0.234,
					"learning_rate":          0.00005,
					"throughput_samples_sec": 128.5,
				},
				EvalMetrics: map[string]interface{}{
					"eval_accuracy": 0.89,
					"eval_f1_score": 0.87,
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
		status    *AnnotationStatus
		wantErr   bool
		checkJSON bool
	}{
		{
			name:     "successful update",
			trainJob: &trainer.TrainJob{},
			status: &AnnotationStatus{
				ProgressPercentage:        ptrInt(10),
				EstimatedRemainingSeconds: ptrInt(9000),
				CurrentStep:               100,
				TotalSteps:                ptrInt(1000),
				CurrentEpoch:              1,
				TotalEpochs:               ptrInt(10),
				LastUpdatedTime:           "2025-11-18T10:00:00Z",
			},
			wantErr:   false,
			checkJSON: true,
		},
		{
			name: "update with metrics",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			status: &AnnotationStatus{
				ProgressPercentage: ptrInt(50),
				CurrentStep:        500,
				TotalSteps:         ptrInt(1000),
				CurrentEpoch:       5,
				TrainMetrics: map[string]interface{}{
					"loss":          0.5,
					"learning_rate": 0.001,
				},
				EvalMetrics: map[string]interface{}{
					"eval_accuracy": 0.95,
					"eval_f1_score": 0.92,
				},
				LastUpdatedTime: "2025-11-18T11:00:00Z",
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

				// Verify metrics are included
				if len(tt.status.TrainMetrics) > 0 {
					metricsMap, ok := decoded["trainMetrics"].(map[string]interface{})
					if !ok {
						t.Error("TrainMetrics not found in decoded JSON")
						return
					}

					for key := range tt.status.TrainMetrics {
						if _, exists := metricsMap[key]; !exists {
							t.Errorf("Train metric %s not found in JSON", key)
						}
					}
				}

				if len(tt.status.EvalMetrics) > 0 {
					metricsMap, ok := decoded["evalMetrics"].(map[string]interface{})
					if !ok {
						t.Error("EvalMetrics not found in decoded JSON")
						return
					}

					for key := range tt.status.EvalMetrics {
						if _, exists := metricsMap[key]; !exists {
							t.Errorf("Eval metric %s not found in JSON", key)
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

func TestValidateTrainerStatus(t *testing.T) {
	tests := []struct {
		name    string
		status  *TrainerStatus
		wantErr bool
	}{
		{
			name: "valid status all fields",
			status: &TrainerStatus{
				ProgressPercentage:        ptrInt(50),
				EstimatedRemainingSeconds: ptrInt(1000),
				CurrentStep:               500,
				TotalSteps:                ptrInt(1000),
				CurrentEpoch:              2,
				TotalEpochs:               ptrInt(5),
			},
			wantErr: false,
		},
		{
			name: "valid status zero progress",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(0),
				CurrentStep:        0,
				CurrentEpoch:       0,
			},
			wantErr: false,
		},
		{
			name: "valid status 100 progress",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(100),
				CurrentStep:        1000,
				TotalSteps:         ptrInt(1000),
			},
			wantErr: false,
		},
		{
			name: "negative progress percentage",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(-1),
				CurrentStep:        500,
			},
			wantErr: true,
		},
		{
			name: "progress percentage over 100",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(101),
				CurrentStep:        500,
			},
			wantErr: true,
		},
		{
			name: "negative current step",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(50),
				CurrentStep:        -1,
			},
			wantErr: true,
		},
		{
			name: "negative total steps",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(50),
				CurrentStep:        500,
				TotalSteps:         ptrInt(-100),
			},
			wantErr: true,
		},
		{
			name: "negative current epoch",
			status: &TrainerStatus{
				ProgressPercentage: ptrInt(50),
				CurrentStep:        500,
				CurrentEpoch:       -1,
			},
			wantErr: true,
		},
		{
			name: "nil progress percentage allowed",
			status: &TrainerStatus{
				ProgressPercentage: nil,
				CurrentStep:        500,
				CurrentEpoch:       1,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTrainerStatus(tt.status)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateTrainerStatus() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestToAnnotationStatus(t *testing.T) {
	tests := []struct {
		name   string
		input  *TrainerStatus
		verify func(*testing.T, *AnnotationStatus)
	}{
		{
			name: "complete status conversion",
			input: &TrainerStatus{
				ProgressPercentage:        ptrInt(45),
				EstimatedRemainingSeconds: ptrInt(3665),
				CurrentStep:               4500,
				TotalSteps:                ptrInt(10000),
				CurrentEpoch:              2,
				TotalEpochs:               ptrInt(5),
				TrainMetrics: map[string]interface{}{
					"loss":          0.234,
					"learning_rate": 0.00005,
				},
				EvalMetrics: map[string]interface{}{
					"eval_accuracy": 0.89,
				},
			},
			verify: func(t *testing.T, result *AnnotationStatus) {
				if result.ProgressPercentage == nil || *result.ProgressPercentage != 45 {
					t.Error("ProgressPercentage not converted correctly")
				}
				if result.EstimatedRemainingSeconds == nil || *result.EstimatedRemainingSeconds != 3665 {
					t.Error("EstimatedRemainingSeconds not converted correctly")
				}
				if result.EstimatedRemainingTimeSummary != "1 hour 1 minute" {
					t.Errorf("EstimatedRemainingTimeSummary = %q, want '1 hour 1 minute'", result.EstimatedRemainingTimeSummary)
				}
				if result.LastUpdatedTime == "" {
					t.Error("LastUpdatedTime should be set")
				}
				if result.CurrentStep != 4500 {
					t.Errorf("CurrentStep = %d, want 4500", result.CurrentStep)
				}
				if len(result.TrainMetrics) != 2 {
					t.Errorf("TrainMetrics length = %d, want 2", len(result.TrainMetrics))
				}
			},
		},
		{
			name: "nil remaining seconds",
			input: &TrainerStatus{
				ProgressPercentage:        ptrInt(50),
				EstimatedRemainingSeconds: nil,
				CurrentStep:               500,
			},
			verify: func(t *testing.T, result *AnnotationStatus) {
				if result.EstimatedRemainingSeconds != nil {
					t.Error("EstimatedRemainingSeconds should be nil")
				}
				if result.EstimatedRemainingTimeSummary != "" {
					t.Errorf("EstimatedRemainingTimeSummary should be empty for nil input, got %q", result.EstimatedRemainingTimeSummary)
				}
			},
		},
		{
			name: "zero remaining seconds",
			input: &TrainerStatus{
				ProgressPercentage:        ptrInt(100),
				EstimatedRemainingSeconds: ptrInt(0),
				CurrentStep:               1000,
			},
			verify: func(t *testing.T, result *AnnotationStatus) {
				if result.EstimatedRemainingTimeSummary != "" {
					t.Errorf("EstimatedRemainingTimeSummary should be empty for zero, got %q", result.EstimatedRemainingTimeSummary)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ToAnnotationStatus(tt.input)
			if result == nil {
				t.Fatal("ToAnnotationStatus returned nil")
			}
			tt.verify(t, result)
		})
	}
}

func TestFormatDurationSummary(t *testing.T) {
	tests := []struct {
		name    string
		seconds *int
		want    string
	}{
		{
			name:    "nil input",
			seconds: nil,
			want:    "",
		},
		{
			name:    "zero or negative",
			seconds: ptrInt(-100),
			want:    "",
		},
		{
			name:    "less than a minute",
			seconds: ptrInt(30),
			want:    "less than a minute",
		},
		{
			name:    "minutes only",
			seconds: ptrInt(90),
			want:    "1 minute",
		},
		{
			name:    "hours only",
			seconds: ptrInt(3600),
			want:    "1 hour",
		},
		{
			name:    "hours and minutes",
			seconds: ptrInt(3665),
			want:    "1 hour 1 minute",
		},
		{
			name:    "days only",
			seconds: ptrInt(86400),
			want:    "1 day",
		},
		{
			name:    "days and hours",
			seconds: ptrInt(90000),
			want:    "1 day 1 hour",
		},
		{
			name:    "truncates to first 2 parts",
			seconds: ptrInt(90060),  // 1 day 1 hour 1 minute
			want:    "1 day 1 hour", // minutes omitted
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatDurationSummary(tt.seconds)
			if got != tt.want {
				t.Errorf("formatDurationSummary() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestIsFinalStatusCaptured(t *testing.T) {
	tests := []struct {
		name     string
		trainJob *trainer.TrainJob
		want     bool
	}{
		{
			name: "nil annotations",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: nil,
				},
			},
			want: false,
		},
		{
			name: "missing trainer status annotation",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"other-annotation": "value",
					},
				},
			},
			want: false,
		},
		{
			name: "progress at 100%",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationTrainerStatus: `{"progressPercentage":100,"currentStep":1000,"totalSteps":1000}`,
					},
				},
			},
			want: true,
		},
		{
			name: "progress incomplete",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationTrainerStatus: `{"progressPercentage":50,"currentStep":500,"totalSteps":1000}`,
					},
				},
			},
			want: false,
		},
		{
			name: "invalid JSON",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationTrainerStatus: `{invalid json}`,
					},
				},
			},
			want: false,
		},
		{
			name: "empty annotation value",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationTrainerStatus: "",
					},
				},
			},
			want: false,
		},
		{
			name: "null progress percentage",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.AnnotationTrainerStatus: `{"progressPercentage":null,"currentStep":500}`,
					},
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsFinalStatusCaptured(tt.trainJob)
			if got != tt.want {
				t.Errorf("IsFinalStatusCaptured() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetPrimaryPod_DifferentLabelPatterns(t *testing.T) {
	tests := []struct {
		name     string
		trainJob *trainer.TrainJob
		pods     []corev1.Pod
		wantErr  bool
		wantPod  string
	}{
		{
			name: "jobset pods with job-index and completion-index",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
			},
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-job-worker-0-0",
						Namespace: "default",
						Labels: map[string]string{
							"jobset.sigs.k8s.io/jobset-name":           "test-job",
							"jobset.sigs.k8s.io/job-index":             "0",
							"batch.kubernetes.io/job-completion-index": "0",
						},
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						PodIP: "10.0.0.1",
					},
				},
			},
			wantErr: false,
			wantPod: "test-job-worker-0-0",
		},
		{
			name: "pytorch worker with replica-index",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pytorch-job",
					Namespace: "default",
				},
			},
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pytorch-job-worker-0",
						Namespace: "default",
						Labels: map[string]string{
							"training.kubeflow.org/job-name":      "pytorch-job",
							"training.kubeflow.org/replica-type":  "Worker",
							"training.kubeflow.org/replica-index": "0",
						},
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						PodIP: "10.0.0.2",
					},
				},
			},
			wantErr: false,
			wantPod: "pytorch-job-worker-0",
		},
		{
			name: "mpi launcher pod",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mpi-job",
					Namespace: "default",
				},
			},
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "mpi-job-launcher",
						Namespace: "default",
						Labels: map[string]string{
							"training.kubeflow.org/job-name":     "mpi-job",
							"training.kubeflow.org/replica-type": "Launcher",
						},
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						PodIP: "10.0.0.3",
					},
				},
			},
			wantErr: false,
			wantPod: "mpi-job-launcher",
		},
		{
			name: "pod without IP returns error",
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
						PodIP: "", // No IP
					},
				},
			},
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

			if !tt.wantErr {
				if pod == nil {
					t.Fatal("GetPrimaryPod() returned nil pod")
				}
				if pod.Name != tt.wantPod {
					t.Errorf("GetPrimaryPod() returned pod %q, want %q", pod.Name, tt.wantPod)
				}
			}
		})
	}
}

// Helper functions
func ptrInt(i int) *int {
	return &i
}

func ptrFloat64(f float64) *float64 {
	return &f
}

func ptrString(s string) *string {
	return &s
}
