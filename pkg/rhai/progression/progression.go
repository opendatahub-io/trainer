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
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
)

var (
	httpClient     *http.Client
	httpClientOnce sync.Once
)

// getHTTPClient returns a shared HTTP client for metrics polling.
func getHTTPClient() *http.Client {
	httpClientOnce.Do(func() {
		httpClient = &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     30 * time.Second,
			},
		}
	})
	return httpClient
}

// TrainerStatus represents the training status from the HTTP metrics endpoint.
type TrainerStatus struct {
	ProgressPercentage        *int                   `json:"progressPercentage"`
	EstimatedRemainingSeconds *int                   `json:"estimatedRemainingSeconds"`
	CurrentStep               int                    `json:"currentStep"`
	TotalSteps                *int                   `json:"totalSteps"`
	CurrentEpoch              int                    `json:"currentEpoch"`
	TotalEpochs               *int                   `json:"totalEpochs"`
	TrainMetrics              map[string]interface{} `json:"trainMetrics"`
	EvalMetrics               map[string]interface{} `json:"evalMetrics"`
}

// AnnotationStatus represents the enhanced format stored in TrainJob annotations.
// Includes training status plus controller-added fields (time summary, lastUpdatedTime).
type AnnotationStatus struct {
	ProgressPercentage            *int                   `json:"progressPercentage"`
	EstimatedRemainingSeconds     *int                   `json:"estimatedRemainingSeconds,omitempty"`
	EstimatedRemainingTimeSummary string                 `json:"estimatedRemainingTimeSummary,omitempty"`
	CurrentStep                   int                    `json:"currentStep"`
	TotalSteps                    *int                   `json:"totalSteps,omitempty"`
	CurrentEpoch                  int                    `json:"currentEpoch"`
	TotalEpochs                   *int                   `json:"totalEpochs,omitempty"`
	TrainMetrics                  map[string]interface{} `json:"trainMetrics,omitempty"`
	EvalMetrics                   map[string]interface{} `json:"evalMetrics,omitempty"`
	LastUpdatedTime               string                 `json:"lastUpdatedTime"`
}

func IsProgressionTrackingEnabled(trainJob *trainer.TrainJob) bool {
	if trainJob.Annotations == nil {
		return false
	}
	enabled, exists := trainJob.Annotations[constants.AnnotationProgressionTracking]
	return exists && enabled == "true"
}

// isPodReady checks if a pod is ready based on its conditions.
func isPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
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

		// Return first running and ready pod with IP from this label set
		for i := range podList.Items {
			pod := &podList.Items[i]
			if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" && isPodReady(pod) {
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

	// Return first running and ready pod with IP
	var podStates []string
	for i := range podList.Items {
		pod := &podList.Items[i]
		if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" && isPodReady(pod) {
			return pod, nil
		}
		// Collect pod states for debugging
		ready := "not ready"
		if isPodReady(pod) {
			ready = "ready"
		}
		podStates = append(podStates, fmt.Sprintf("%s: %s (IP: %s, %s)", pod.Name, pod.Status.Phase, pod.Status.PodIP, ready))
	}

	return nil, fmt.Errorf("no running and ready pod with IP found for TrainJob %s/%s; found pods: %v", trainJob.Namespace, trainJob.Name, podStates)
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
	requestCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	resp, err := getHTTPClient().Do(req)
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

	var status TrainerStatus
	if err := json.Unmarshal(body, &status); err != nil {
		return nil, fmt.Errorf("failed to parse metrics JSON: %w", err)
	}

	if err := validateTrainerStatus(&status); err != nil {
		return nil, fmt.Errorf("invalid training metrics: %w", err)
	}

	return &status, nil
}

// validateTrainerStatus validates training metrics are within acceptable bounds.
func validateTrainerStatus(m *TrainerStatus) error {
	if m.ProgressPercentage != nil && (*m.ProgressPercentage < 0 || *m.ProgressPercentage > 100) {
		return fmt.Errorf("progressPercentage must be between 0 and 100, got %d", *m.ProgressPercentage)
	}
	if m.CurrentStep < 0 {
		return fmt.Errorf("currentStep must be non-negative, got %d", m.CurrentStep)
	}
	if m.TotalSteps != nil && *m.TotalSteps < 0 {
		return fmt.Errorf("totalSteps must be non-negative, got %d", *m.TotalSteps)
	}
	if m.CurrentEpoch < 0 {
		return fmt.Errorf("currentEpoch must be non-negative, got %d", m.CurrentEpoch)
	}
	return nil
}

// ToAnnotationStatus converts training status to annotation format with enhancements.
func ToAnnotationStatus(status *TrainerStatus) *AnnotationStatus {
	return &AnnotationStatus{
		ProgressPercentage:            status.ProgressPercentage,
		EstimatedRemainingSeconds:     status.EstimatedRemainingSeconds,
		EstimatedRemainingTimeSummary: formatDurationSummary(status.EstimatedRemainingSeconds),
		CurrentStep:                   status.CurrentStep,
		TotalSteps:                    status.TotalSteps,
		CurrentEpoch:                  status.CurrentEpoch,
		TotalEpochs:                   status.TotalEpochs,
		TrainMetrics:                  status.TrainMetrics,
		EvalMetrics:                   status.EvalMetrics,
		LastUpdatedTime:               time.Now().UTC().Format(time.RFC3339),
	}
}

// formatDurationSummary converts seconds to human-readable format (e.g., "9 days 5 hours").
func formatDurationSummary(seconds *int) string {
	if seconds == nil || *seconds <= 0 {
		return ""
	}

	s := *seconds
	days := s / 86400
	hours := (s % 86400) / 3600
	minutes := (s % 3600) / 60

	var parts []string
	if days > 0 {
		parts = append(parts, fmt.Sprintf("%d day%s", days, plural(days)))
	}
	if hours > 0 {
		parts = append(parts, fmt.Sprintf("%d hour%s", hours, plural(hours)))
	}
	if minutes > 0 && days == 0 { // Only show minutes if no days
		parts = append(parts, fmt.Sprintf("%d minute%s", minutes, plural(minutes)))
	}

	if len(parts) == 0 {
		return "less than a minute"
	}

	// Join first 2 parts
	if len(parts) > 2 {
		parts = parts[:2]
	}

	result := parts[0]
	if len(parts) > 1 {
		result += " " + parts[1]
	}
	return result
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func UpdateTrainerStatusAnnotation(trainJob *trainer.TrainJob, status *AnnotationStatus) error {
	if trainJob.Annotations == nil {
		trainJob.Annotations = make(map[string]string)
	}

	statusJSON, err := json.Marshal(status)
	if err != nil {
		return fmt.Errorf("failed to marshal controller status: %w", err)
	}

	trainJob.Annotations[constants.AnnotationTrainerStatus] = string(statusJSON)
	return nil
}

func PollAndUpdateProgress(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	pod, err := GetPrimaryPod(ctx, c, trainJob)
	if err != nil {
		return false, fmt.Errorf("primary pod not available: %w", err)
	}

	metricsPort := GetMetricsPort(trainJob)
	status, err := PollTrainingProgress(ctx, pod, metricsPort)
	if err != nil {
		return false, fmt.Errorf("failed to poll metrics: %w", err)
	}

	annotationStatus := ToAnnotationStatus(status)
	if err := UpdateTrainerStatusAnnotation(trainJob, annotationStatus); err != nil {
		return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
	}

	if err := c.Update(ctx, trainJob); err != nil {
		return false, fmt.Errorf("failed to update TrainJob: %w", err)
	}

	return true, nil
}

func IsFinalStatusCaptured(trainJob *trainer.TrainJob) bool {
	if trainJob.Annotations == nil {
		return false
	}

	statusJSON, exists := trainJob.Annotations[constants.AnnotationTrainerStatus]
	if !exists || statusJSON == "" {
		return false
	}

	var status AnnotationStatus
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return false
	}

	return status.ProgressPercentage != nil && *status.ProgressPercentage == 100
}

func PollAndUpdateFinalProgress(ctx context.Context, c client.Client, trainJob *trainer.TrainJob, completed bool) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	// Try to get final metrics from pod if it still exists
	var annotationStatus *AnnotationStatus

	pod, err := GetPrimaryPod(ctx, c, trainJob)
	if err == nil {
		metricsPort := GetMetricsPort(trainJob)
		if status, pollErr := PollTrainingProgress(ctx, pod, metricsPort); pollErr == nil {
			annotationStatus = ToAnnotationStatus(status)
		}
	}

	// If we couldn't get metrics, synthesize final status
	if annotationStatus == nil {
		progress := 100
		remaining := 0
		annotationStatus = &AnnotationStatus{
			ProgressPercentage:            &progress,
			EstimatedRemainingSeconds:     &remaining,
			EstimatedRemainingTimeSummary: "complete",
			LastUpdatedTime:               time.Now().UTC().Format(time.RFC3339),
		}

		// Try to preserve existing metrics if available
		if existingStatus := getExistingStatus(trainJob); existingStatus != nil {
			annotationStatus.CurrentStep = existingStatus.CurrentStep
			annotationStatus.TotalSteps = existingStatus.TotalSteps
			annotationStatus.CurrentEpoch = existingStatus.CurrentEpoch
			annotationStatus.TotalEpochs = existingStatus.TotalEpochs
			annotationStatus.TrainMetrics = existingStatus.TrainMetrics
			annotationStatus.EvalMetrics = existingStatus.EvalMetrics
		}
	}

	// Ensure final status shows 100%
	progress := 100
	remaining := 0
	annotationStatus.ProgressPercentage = &progress
	annotationStatus.EstimatedRemainingSeconds = &remaining
	if annotationStatus.EstimatedRemainingTimeSummary == "" {
		annotationStatus.EstimatedRemainingTimeSummary = "complete"
	}
	annotationStatus.LastUpdatedTime = time.Now().UTC().Format(time.RFC3339)

	if err := UpdateTrainerStatusAnnotation(trainJob, annotationStatus); err != nil {
		return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
	}

	if err := c.Update(ctx, trainJob); err != nil {
		return false, fmt.Errorf("failed to update TrainJob: %w", err)
	}

	return true, nil
}

func getExistingStatus(trainJob *trainer.TrainJob) *AnnotationStatus {
	if trainJob.Annotations == nil {
		return nil
	}

	statusJSON, exists := trainJob.Annotations[constants.AnnotationTrainerStatus]
	if !exists || statusJSON == "" {
		return nil
	}

	var status AnnotationStatus
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return nil
	}

	return &status
}
