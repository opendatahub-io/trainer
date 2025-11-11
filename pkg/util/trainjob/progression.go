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
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
)

// TrainerStatus represents the complete training status from the metrics endpoint
type TrainerStatus struct {
	Status        string        `json:"status"`
	StatusMessage string        `json:"status_message,omitempty"`
	StatusDetails StatusDetails `json:"status_details,omitempty"`
	Progress      Progress      `json:"progress,omitempty"`
	Time          TimeMetrics   `json:"time,omitempty"`
	Metrics       Metrics       `json:"metrics,omitempty"`
	Checkpoint    Checkpoint    `json:"checkpoint,omitempty"`
}

// StatusDetails contains event tracking and warnings
type StatusDetails struct {
	LastEvent     string    `json:"last_event,omitempty"`
	LastEventTime string    `json:"last_event_time,omitempty"`
	Warnings      []Warning `json:"warnings,omitempty"`
}

// Warning represents a warning message with timestamp
type Warning struct {
	Time    string `json:"time"`
	Message string `json:"message"`
}

// Progress contains training progress metrics
type Progress struct {
	StepCurrent int     `json:"step_current"`
	StepTotal   int     `json:"step_total,omitempty"`
	Percent     float64 `json:"percent"`
	Epoch       int     `json:"epoch"`
}

// TimeMetrics contains time-related metrics
type TimeMetrics struct {
	ElapsedSec   float64  `json:"elapsed_sec"`
	Elapsed      string   `json:"elapsed"`
	RemainingSec *float64 `json:"remaining_sec,omitempty"`
	Remaining    *string  `json:"remaining,omitempty"`
	StartedSec   float64  `json:"started_sec,omitempty"`
	StartedAt    string   `json:"started_at,omitempty"`
	UpdatedSec   float64  `json:"updated_sec,omitempty"`
	UpdatedAt    string   `json:"updated_at,omitempty"`
}

// Metrics contains training metrics (loss, lr, custom metrics, etc.)
type Metrics struct {
	Loss                 *float64               `json:"loss,omitempty"`
	LearningRate         *float64               `json:"learning_rate,omitempty"`
	ThroughputSamplesSec *float64               `json:"throughput_samples_sec,omitempty"`
	Custom               map[string]interface{} `json:"-"` // For custom metrics
}

// Checkpoint contains checkpoint-related information
type Checkpoint struct {
	ResumedFrom string `json:"resumed_from,omitempty"`
	LastStep    int    `json:"last_step,omitempty"`
	LastPath    string `json:"last_path,omitempty"`
	SavedAt     string `json:"saved_at,omitempty"`
}

func IsProgressionTrackingEnabled(trainJob *trainer.TrainJob) bool {
	if trainJob.Annotations == nil {
		return false
	}
	enabled, exists := trainJob.Annotations[constants.AnnotationProgressionTracking]
	return exists && enabled == "enabled"
}

func GetPrimaryPod(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) (*corev1.Pod, error) {
	// First, try to find the rank 0 pod (primary worker or launcher)
	podList := &corev1.PodList{}

	// Try common label patterns for primary pod (rank 0)
	labelSets := []labels.Set{
		// JobSet-created pods (new v2 API with TrainingRuntime)
		{
			"jobset.sigs.k8s.io/jobset-name":           trainJob.Name,
			"jobset.sigs.k8s.io/job-index":             "0",
			"batch.kubernetes.io/job-completion-index": "0",
		},
		// JobSet fallback: just jobset name
		{
			"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
		},
		// PyTorch/TensorFlow worker with replica-index (legacy training-operator v1)
		{
			"training.kubeflow.org/job-name":      trainJob.Name,
			"training.kubeflow.org/replica-type":  "Worker",
			"training.kubeflow.org/replica-index": "0",
		},
		// MPI launcher (legacy training-operator v1)
		{
			"training.kubeflow.org/job-name":     trainJob.Name,
			"training.kubeflow.org/replica-type": "Launcher",
		},
		// Fallback: just job name with replica-index 0 (legacy)
		{
			"training.kubeflow.org/job-name":      trainJob.Name,
			"training.kubeflow.org/replica-index": "0",
		},
	}

	// Try each label selector pattern
	for _, labelSet := range labelSets {
		labelSelector := labels.SelectorFromSet(labelSet)
		if err := c.List(ctx, podList, &client.ListOptions{
			Namespace:     trainJob.Namespace,
			LabelSelector: labelSelector,
		}); err != nil {
			return nil, fmt.Errorf("failed to list pods: %w", err)
		}

		// Return first running pod with IP from this label set
		for i := range podList.Items {
			pod := &podList.Items[i]
			if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
				return pod, nil
			}
		}
	}

	// Fallback: find any running pod with old training.kubeflow.org labels
	labelSelector := labels.SelectorFromSet(labels.Set{
		"training.kubeflow.org/job-name": trainJob.Name,
	})
	if err := c.List(ctx, podList, &client.ListOptions{
		Namespace:     trainJob.Namespace,
		LabelSelector: labelSelector,
	}); err != nil {
		return nil, fmt.Errorf("failed to list pods: %w", err)
	}

	// Final fallback: find any running pod with new jobset labels
	if len(podList.Items) == 0 {
		labelSelector = labels.SelectorFromSet(labels.Set{
			"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
		})
		if err := c.List(ctx, podList, &client.ListOptions{
			Namespace:     trainJob.Namespace,
			LabelSelector: labelSelector,
		}); err != nil {
			return nil, fmt.Errorf("failed to list pods: %w", err)
		}
	}

	if len(podList.Items) == 0 {
		return nil, fmt.Errorf("no pods found for TrainJob %s/%s", trainJob.Namespace, trainJob.Name)
	}

	// Return first running pod with IP
	for i := range podList.Items {
		pod := &podList.Items[i]
		if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
			return pod, nil
		}
	}

	return nil, fmt.Errorf("no running pod with IP found for TrainJob %s/%s", trainJob.Namespace, trainJob.Name)
}

func GetMetricsPort(trainJob *trainer.TrainJob) string {
	if trainJob.Annotations == nil {
		return constants.DefaultMetricsPort
	}
	if port, exists := trainJob.Annotations[constants.AnnotationMetricsPort]; exists && port != "" {
		return port
	}
	return constants.DefaultMetricsPort
}

// GetMetricsPollInterval returns the metrics polling interval for a TrainJob.
// The interval can be configured via trainer.odh.org/metrics-poll-interval annotation.
// Supports formats: "30" (seconds), "30s", "1m", etc.
// Min: 5s, Max: 300s (5 minutes), Default: 30s
func GetMetricsPollInterval(trainJob *trainer.TrainJob) time.Duration {
	defaultInterval := time.Duration(constants.DefaultMetricsPollIntervalSecs) * time.Second

	if trainJob.Annotations == nil {
		return defaultInterval
	}

	intervalStr, exists := trainJob.Annotations[constants.AnnotationMetricsPollInterval]
	if !exists || intervalStr == "" {
		return defaultInterval
	}

	// Parse the interval string (supports duration formats like "30s", "1m", etc.)
	interval, err := time.ParseDuration(intervalStr)
	if err != nil {
		// Try parsing as integer seconds if no unit provided (e.g., "30" means 30 seconds)
		if seconds, parseErr := strconv.Atoi(intervalStr); parseErr == nil {
			interval = time.Duration(seconds) * time.Second
		} else {
			// If parsing fails completely, return default
			return defaultInterval
		}
	}

	// Enforce min/max bounds (5s - 300s)
	if interval < 5*time.Second {
		return 5 * time.Second
	}
	if interval > 300*time.Second {
		return 300 * time.Second
	}

	return interval
}

func PollTrainingProgress(ctx context.Context, pod *corev1.Pod, metricsPort string) (*TrainerStatus, error) {
	if pod.Status.PodIP == "" {
		return nil, fmt.Errorf("pod %s/%s has no IP address", pod.Namespace, pod.Name)
	}

	metricsURL := fmt.Sprintf("http://%s:%s/metrics", pod.Status.PodIP, metricsPort)
	httpClient := &http.Client{Timeout: 5 * time.Second}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch metrics from %s: %w", metricsURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code %d from metrics endpoint", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// First unmarshal into a map to handle custom metrics
	var rawStatus map[string]interface{}
	if err := json.Unmarshal(body, &rawStatus); err != nil {
		return nil, fmt.Errorf("failed to parse metrics JSON: %w", err)
	}

	// Then unmarshal into the typed structure
	var status TrainerStatus
	if err := json.Unmarshal(body, &status); err != nil {
		return nil, fmt.Errorf("failed to parse metrics JSON into struct: %w", err)
	}

	// Extract custom metrics (any metrics beyond the standard ones)
	if metricsRaw, ok := rawStatus["metrics"].(map[string]interface{}); ok {
		status.Metrics.Custom = make(map[string]interface{})
		for key, value := range metricsRaw {
			// Skip standard metrics, store everything else as custom
			if key != "loss" && key != "learning_rate" && key != "throughput_samples_sec" {
				status.Metrics.Custom[key] = value
			}
		}
	}

	return &status, nil
}

func UpdateTrainerStatusAnnotation(trainJob *trainer.TrainJob, status *TrainerStatus) error {
	if trainJob.Annotations == nil {
		trainJob.Annotations = make(map[string]string)
	}

	// Marshal to JSON, handling custom metrics
	statusMap := make(map[string]interface{})

	// Marshal the struct first
	statusBytes, err := json.Marshal(status)
	if err != nil {
		return fmt.Errorf("failed to marshal trainer status: %w", err)
	}

	// Unmarshal into map to add custom metrics
	if err := json.Unmarshal(statusBytes, &statusMap); err != nil {
		return fmt.Errorf("failed to unmarshal status to map: %w", err)
	}

	// Add custom metrics if they exist
	if len(status.Metrics.Custom) > 0 {
		if metricsMap, ok := statusMap["metrics"].(map[string]interface{}); ok {
			for key, value := range status.Metrics.Custom {
				metricsMap[key] = value
			}
		}
	}

	// Marshal the final map
	finalJSON, err := json.Marshal(statusMap)
	if err != nil {
		return fmt.Errorf("failed to marshal final status: %w", err)
	}

	trainJob.Annotations[constants.AnnotationTrainerStatus] = string(finalJSON)
	return nil
}

func PollAndUpdateProgress(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	pod, err := GetPrimaryPod(ctx, c, trainJob)
	if err != nil {
		// Pod might not be ready yet - return nil error but log the reason
		return false, fmt.Errorf("primary pod not available: %w", err)
	}

	metricsPort := GetMetricsPort(trainJob)

	status, err := PollTrainingProgress(ctx, pod, metricsPort)
	if err != nil {
		// Metrics server might not be ready yet - return error for visibility
		return false, fmt.Errorf("failed to poll metrics: %w", err)
	}

	if err := UpdateTrainerStatusAnnotation(trainJob, status); err != nil {
		return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
	}

	if err := c.Update(ctx, trainJob); err != nil {
		return false, fmt.Errorf("failed to update TrainJob: %w", err)
	}

	return true, nil
}

func IsFinalStatusCaptured(trainJob *trainer.TrainJob) bool {
	// Check if status is already "completed" or "failed"
	existingStatus, err := parseExistingTrainerStatus(trainJob)
	if err != nil {
		return false
	}
	return existingStatus.Status == "completed" || existingStatus.Status == "failed"
}

func PollAndUpdateFinalProgress(ctx context.Context, c client.Client, trainJob *trainer.TrainJob, completed bool) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	// Try to get final metrics from pod if it still exists
	pod, err := GetPrimaryPod(ctx, c, trainJob)
	var status *TrainerStatus

	if err == nil {
		// Pod still exists, try to get final metrics
		metricsPort := GetMetricsPort(trainJob)
		status, err = PollTrainingProgress(ctx, pod, metricsPort)
	}

	// If we couldn't get metrics from pod (pod terminated, metrics server down, etc.),
	// synthesize final status based on job completion state
	if err != nil || status == nil {
		status = &TrainerStatus{
			Progress: Progress{
				Percent: 100.0,
			},
		}

		if completed {
			status.Status = "completed"
			status.StatusMessage = "Training completed successfully"
		} else {
			status.Status = "failed"
			status.StatusMessage = "Training failed"
		}

		// Try to preserve some existing metrics if available
		if existingStatus, parseErr := parseExistingTrainerStatus(trainJob); parseErr == nil && existingStatus != nil {
			// Preserve existing metrics and time data
			status.Metrics = existingStatus.Metrics
			status.Time = existingStatus.Time
			status.Checkpoint = existingStatus.Checkpoint
			// Update progress to 100% and set final status
			status.Progress.StepCurrent = existingStatus.Progress.StepTotal
			if status.Progress.StepCurrent == 0 {
				status.Progress.StepCurrent = existingStatus.Progress.StepCurrent
			}
			status.Progress.Epoch = existingStatus.Progress.Epoch
		}
	} else {
		// Got metrics from pod - ensure it's marked as completed/failed
		if completed {
			status.Status = "completed"
			status.Progress.Percent = 100.0
			if status.StatusMessage == "" || status.Status == "training" {
				status.StatusMessage = "Training completed successfully"
			}
		} else {
			status.Status = "failed"
			if status.StatusMessage == "" || status.Status == "training" {
				status.StatusMessage = "Training failed"
			}
		}
	}

	// Update annotation with final status
	if err := UpdateTrainerStatusAnnotation(trainJob, status); err != nil {
		return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
	}

	if err := c.Update(ctx, trainJob); err != nil {
		return false, fmt.Errorf("failed to update TrainJob: %w", err)
	}

	return true, nil
}

func parseExistingTrainerStatus(trainJob *trainer.TrainJob) (*TrainerStatus, error) {
	if trainJob.Annotations == nil {
		return nil, fmt.Errorf("no annotations")
	}

	statusJSON, exists := trainJob.Annotations[constants.AnnotationTrainerStatus]
	if !exists || statusJSON == "" {
		return nil, fmt.Errorf("no trainer status annotation")
	}

	var status TrainerStatus
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return nil, fmt.Errorf("failed to parse trainer status: %w", err)
	}

	return &status, nil
}
