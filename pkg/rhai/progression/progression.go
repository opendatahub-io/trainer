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

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
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
	CurrentStep               *int                   `json:"currentStep"`
	TotalSteps                *int                   `json:"totalSteps"`
	CurrentEpoch              *float64               `json:"currentEpoch"` // float64 for precision (1.98 not 1)
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
	CurrentStep                   *int                   `json:"currentStep,omitempty"`
	TotalSteps                    *int                   `json:"totalSteps,omitempty"`
	CurrentEpoch                  *float64               `json:"currentEpoch,omitempty"` // float64 for precision (1.98 not 1)
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

// GetPrimaryPod returns the first running and ready pod with an IP for a TrainJob.
// Uses the provided reader (typically APIReader) to avoid setting up pod informers/watchers.
func GetPrimaryPod(ctx context.Context, reader client.Reader, trainJob *trainer.TrainJob) (*corev1.Pod, error) {
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
		if err := reader.List(ctx, podList, &client.ListOptions{
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
	if err := reader.List(ctx, podList, &client.ListOptions{
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
		if err := reader.List(ctx, podList, &client.ListOptions{
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
	requestCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
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

	cleanInvalidMetrics(&status)

	return &status, nil
}

// cleanInvalidMetrics removes invalid values while keeping valid fields.
// Defense against custom implementations, malformed requests, or edge cases.
func cleanInvalidMetrics(m *TrainerStatus) {
	if m.ProgressPercentage != nil && (*m.ProgressPercentage < 0 || *m.ProgressPercentage > 100) {
		m.ProgressPercentage = nil
	}
	if m.EstimatedRemainingSeconds != nil && *m.EstimatedRemainingSeconds < 0 {
		m.EstimatedRemainingSeconds = nil
	}
	if m.TotalSteps != nil && *m.TotalSteps < 0 {
		m.TotalSteps = nil
	}
	if m.TotalEpochs != nil && *m.TotalEpochs < 0 {
		m.TotalEpochs = nil
	}
	if m.CurrentStep != nil && *m.CurrentStep < 0 {
		zero := 0
		m.CurrentStep = &zero
	}
	if m.CurrentEpoch != nil && *m.CurrentEpoch < 0 {
		zeroEpoch := 0.0
		m.CurrentEpoch = &zeroEpoch
	}
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

func PollAndUpdateProgress(ctx context.Context, c client.Client, reader client.Reader, trainJob *trainer.TrainJob) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	pod, err := GetPrimaryPod(ctx, reader, trainJob)
	if err != nil {
		return false, fmt.Errorf("primary pod not available: %w", err)
	}

	metricsPort := GetMetricsPort(trainJob)
	status, err := PollTrainingProgress(ctx, pod, metricsPort)
	if err != nil {
		return false, fmt.Errorf("failed to poll metrics: %w", err)
	}

	annotationStatus := ToAnnotationStatus(status)

	// Use Patch to avoid conflicts with main controller
	patch := client.MergeFrom(trainJob.DeepCopy())
	if err := UpdateTrainerStatusAnnotation(trainJob, annotationStatus); err != nil {
		return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
	}
	if err := c.Patch(ctx, trainJob, patch); err != nil {
		return false, fmt.Errorf("failed to patch TrainJob annotations: %w", err)
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

	// Consider final status captured if:
	// 1. We have a progress percentage (any value, including <100% for epoch-based training)
	// 2. Estimated remaining time is 0 (indicates training has ended)
	if status.ProgressPercentage == nil {
		return false
	}

	// Check if remaining time is explicitly 0 or summary indicates completion
	hasZeroRemaining := status.EstimatedRemainingSeconds != nil && *status.EstimatedRemainingSeconds == 0
	hasCompleteSummary := status.EstimatedRemainingTimeSummary == "complete" ||
		status.EstimatedRemainingTimeSummary == "0 seconds"

	if hasZeroRemaining || hasCompleteSummary {
		return true
	}

	// Also consider captured if job completed/failed AND preStop window has expired
	// This handles cases where on_train_end() never fired (pod killed, crash, etc.)
	// Check if we're past the preStop hook duration since last update
	if status.LastUpdatedTime != "" {
		lastUpdate, err := time.Parse(time.RFC3339, status.LastUpdatedTime)
		if err == nil {
			pollInterval := GetMetricsPollInterval(trainJob)
			preStopDuration := pollInterval*2 + time.Duration(constants.PreStopBufferSecs)*time.Second
			gracePeriod := preStopDuration + time.Duration(constants.TerminationGraceBufferSecs)*time.Second

			// If it's been longer than preStop + grace period, pod is definitely gone
			if time.Since(lastUpdate) > gracePeriod {
				return true // Stop trying, preserve last known state
			}
		}
	}

	return false
}

func PollAndUpdateFinalProgress(ctx context.Context, c client.Client, reader client.Reader, trainJob *trainer.TrainJob, completed bool) (bool, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return false, nil
	}

	// Try to get final metrics from pod if it still exists
	pod, err := GetPrimaryPod(ctx, reader, trainJob)
	if err == nil {
		metricsPort := GetMetricsPort(trainJob)
		if status, pollErr := PollTrainingProgress(ctx, pod, metricsPort); pollErr == nil {
			// Got real metrics from pod - update with final status
			annotationStatus := ToAnnotationStatus(status)
			annotationStatus.LastUpdatedTime = time.Now().UTC().Format(time.RFC3339)

			// Add descriptive summary
			if completed {
				// Detect early stop: currentStep < totalSteps
				earlyStop := false
				if annotationStatus.CurrentStep != nil && annotationStatus.TotalSteps != nil && *annotationStatus.TotalSteps > 0 {
					if *annotationStatus.CurrentStep < *annotationStatus.TotalSteps {
						earlyStop = true
					}
				}
				if earlyStop {
					annotationStatus.EstimatedRemainingTimeSummary = "complete (early stopped)"
				} else {
					annotationStatus.EstimatedRemainingTimeSummary = "complete"
				}
			} else {
				// For failed jobs: show progress context in summary
				progressPct := 0
				if annotationStatus.ProgressPercentage != nil {
					progressPct = *annotationStatus.ProgressPercentage
				}
				annotationStatus.EstimatedRemainingTimeSummary = fmt.Sprintf("failed at %d%%", progressPct)
			}

			// Use Patch to avoid conflicts with main controller
			patch := client.MergeFrom(trainJob.DeepCopy())
			if err := UpdateTrainerStatusAnnotation(trainJob, annotationStatus); err != nil {
				return false, fmt.Errorf("failed to update trainer status annotation: %w", err)
			}
			if err := c.Patch(ctx, trainJob, patch); err != nil {
				return false, fmt.Errorf("failed to patch TrainJob annotations: %w", err)
			}

			return true, nil
		}
	}

	// Pod not available - update final status using existing metrics
	// For completed: force remaining time to 0 (no work remains)
	// For failed: keep remaining time estimate (useful for resume)
	// Use Patch to avoid conflicts with main controller
	patch := client.MergeFrom(trainJob.DeepCopy())
	if err := updateFinalStatus(trainJob, completed); err != nil {
		return false, fmt.Errorf("failed to update final status: %w", err)
	}
	if err := c.Patch(ctx, trainJob, patch); err != nil {
		return false, fmt.Errorf("failed to patch TrainJob annotations: %w", err)
	}

	return true, nil
}

func updateFinalStatus(trainJob *trainer.TrainJob, completed bool) error {
	if trainJob.Annotations == nil {
		return nil // No existing status to update
	}

	statusJSON, exists := trainJob.Annotations[constants.AnnotationTrainerStatus]
	if !exists || statusJSON == "" {
		return nil // No existing status to update
	}

	var status AnnotationStatus
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return err
	}

	// Only update summary - don't modify actual metrics (progress %, remaining time, etc.)
	if completed {
		// Detect early stop: currentStep < totalSteps
		earlyStop := false
		if status.CurrentStep != nil && status.TotalSteps != nil && *status.TotalSteps > 0 {
			if *status.CurrentStep < *status.TotalSteps {
				earlyStop = true
			}
		}
		if earlyStop {
			status.EstimatedRemainingTimeSummary = "early stopped"
		} else {
			status.EstimatedRemainingTimeSummary = "complete"
		}
	} else {
		// For failed jobs: show progress context in summary
		progressPct := 0
		if status.ProgressPercentage != nil {
			progressPct = *status.ProgressPercentage
		}
		status.EstimatedRemainingTimeSummary = fmt.Sprintf("failed at %d%%", progressPct)
	}
	status.LastUpdatedTime = time.Now().UTC().Format(time.RFC3339)

	return UpdateTrainerStatusAnnotation(trainJob, &status)
}

// InjectPreStopHookToApplyConfig adds a preStop lifecycle hook to the primary pod container
// using Apply Configuration. This is used by the jobset plugin to inject the hook during pod creation.
func InjectPreStopHookToApplyConfig(podSpecAC *corev1ac.PodSpecApplyConfiguration, trainJob *trainer.TrainJob) error {
	if !IsProgressionTrackingEnabled(trainJob) || podSpecAC == nil {
		return nil
	}

	if len(podSpecAC.Containers) == 0 {
		return fmt.Errorf("no containers in pod spec")
	}

	// Calculate preStop duration based on poll interval
	pollInterval := GetMetricsPollInterval(trainJob)
	preStopDuration := pollInterval*2 + time.Duration(constants.PreStopBufferSecs)*time.Second
	preStopSleep := int(preStopDuration.Seconds())

	// Termination grace must be greater than preStop duration
	terminationGrace := int64((preStopDuration + time.Duration(constants.TerminationGraceBufferSecs)*time.Second).Seconds())

	// Find the primary trainer container by name (typically "node")
	containerIdx := -1
	for i, container := range podSpecAC.Containers {
		if container.Name != nil && *container.Name == "node" {
			containerIdx = i
			break
		}
	}

	// Fallback to first container if "node" not found
	if containerIdx == -1 {
		containerIdx = 0
	}

	// Inject preStop hook into the target container
	lifecycle := corev1ac.Lifecycle().
		WithPreStop(corev1ac.LifecycleHandler().
			WithExec(corev1ac.ExecAction().
				WithCommand("sleep", strconv.Itoa(preStopSleep))))

	podSpecAC.Containers[containerIdx].WithLifecycle(lifecycle)

	// Set termination grace period (use max of existing and calculated)
	if podSpecAC.TerminationGracePeriodSeconds == nil ||
		*podSpecAC.TerminationGracePeriodSeconds < terminationGrace {
		podSpecAC.WithTerminationGracePeriodSeconds(terminationGrace)
	}

	return nil
}

// InjectPreStopHook adds a preStop lifecycle hook to the primary pod container.
// The hook keeps the metrics server alive after training completes, ensuring
// the controller can capture final metrics before pod termination.
//
// PreStop duration is calculated as: (2 × poll_interval) + buffer
// This guarantees at least 2 poll opportunities after training completion.
func InjectPreStopHook(podSpec *corev1.PodSpec, trainJob *trainer.TrainJob) error {
	if !IsProgressionTrackingEnabled(trainJob) {
		return nil
	}

	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("no containers in pod spec")
	}

	// Inject into primary container (index 0)
	container := &podSpec.Containers[0]

	// Initialize lifecycle if nil
	if container.Lifecycle == nil {
		container.Lifecycle = &corev1.Lifecycle{}
	}

	// Calculate preStop duration based on poll interval
	pollInterval := GetMetricsPollInterval(trainJob)
	preStopDuration := pollInterval*2 + time.Duration(constants.PreStopBufferSecs)*time.Second
	preStopSleep := int(preStopDuration.Seconds())

	// Add preStop hook
	container.Lifecycle.PreStop = &corev1.LifecycleHandler{
		Exec: &corev1.ExecAction{
			Command: []string{"sleep", strconv.Itoa(preStopSleep)},
		},
	}

	// Set termination grace period (must be > preStop duration)
	terminationGrace := preStopDuration + time.Duration(constants.TerminationGraceBufferSecs)*time.Second
	terminationGraceSecs := int64(terminationGrace.Seconds())
	podSpec.TerminationGracePeriodSeconds = &terminationGraceSecs

	return nil
}

// ReconcileProgression handles progression tracking during TrainJob reconciliation.
// Returns ctrl.Result for requeue behavior and any errors encountered.
// This should be called at the end of TrainJob reconciliation when progression tracking is enabled.
func ReconcileProgression(ctx context.Context, c client.Client, reader client.Reader, log logr.Logger, trainJob *trainer.TrainJob) (ctrl.Result, error) {
	if !IsProgressionTrackingEnabled(trainJob) {
		return ctrl.Result{}, nil
	}

	isRunning := !meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobSuspended) &&
		!meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobComplete) &&
		!meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobFailed)

	isCompleted := meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobComplete)
	isFailed := meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobFailed)

	if isRunning {
		// Poll metrics while job is running
		if _, pollErr := PollAndUpdateProgress(ctx, c, reader, trainJob); pollErr != nil {
			log.V(1).Info("Failed to poll training progress", "error", pollErr)
		} else {
			log.V(2).Info("Successfully updated training progress")
		}
		// Requeue to continue polling while job is running
		pollInterval := GetMetricsPollInterval(trainJob)
		log.V(2).Info("Requeuing for metrics polling", "interval", pollInterval)
		return ctrl.Result{RequeueAfter: pollInterval}, nil
	}

	if (isCompleted || isFailed) && !IsFinalStatusCaptured(trainJob) {
		// Job just completed/failed - capture final metrics
		// PreStop hook keeps pod alive, so this should succeed
		captured, pollErr := PollAndUpdateFinalProgress(ctx, c, reader, trainJob, isCompleted)
		if pollErr != nil {
			log.V(1).Info("Failed to capture final training progress, will retry", "error", pollErr, "completed", isCompleted)
			// Requeue quickly - pod should still be alive in preStop window
			return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
		}
		if !captured {
			log.V(1).Info("Pod not available for final metrics poll, will retry", "completed", isCompleted)
			// Pod might be in preStop or already terminated, retry a few times
			return ctrl.Result{RequeueAfter: 2 * time.Second}, nil
		}
		log.Info("Captured final training progress", "completed", isCompleted)
	}

	return ctrl.Result{}, nil
}
