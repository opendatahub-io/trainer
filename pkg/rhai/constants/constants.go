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

// Package constants contains shared constants for all RHAI features.
package constants

const (
	// Progression tracking feature annotations

	// AnnotationProgressionTracking enables/disables progression tracking for a TrainJob.
	// Value: "true" to enable tracking, any other value or absence disables it.
	// Example: trainer.opendatahub.io/progression-tracking: "true"
	AnnotationProgressionTracking string = "trainer.opendatahub.io/progression-tracking"

	// AnnotationTrainerStatus stores the JSON-encoded training status/progress.
	// This annotation is automatically updated by the controller with real-time metrics.
	// Example: trainer.opendatahub.io/trainerStatus: '{"status":"training","progress":{"percent":45.2},...}'
	AnnotationTrainerStatus string = "trainer.opendatahub.io/trainerStatus"

	// AnnotationMetricsPort specifies the port where the training pod exposes metrics.
	// Default: 28080. Valid range: 1024-65535 (non-privileged ports).
	// Ports 0-1023 require root privileges and are incompatible with OpenShift
	// restricted SCCs and Kubernetes non-root security policies.
	// Example: trainer.opendatahub.io/metrics-port: "8080"
	AnnotationMetricsPort string = "trainer.opendatahub.io/metrics-port"

	// AnnotationMetricsPollInterval specifies how often to poll metrics (supports duration format).
	// Accepts: "30s", "1m", or integer seconds "30" (min: 5s, max: 300s)
	// Default: 30s
	// Example: trainer.opendatahub.io/metrics-poll-interval: "45s"
	AnnotationMetricsPollInterval string = "trainer.opendatahub.io/metrics-poll-interval"

	// DefaultMetricsPort is the default port for metrics endpoints in training pods.
	DefaultMetricsPort string = "28080"

	// DefaultMetricsPollIntervalSecs is the default interval (in seconds) for polling training metrics.
	DefaultMetricsPollIntervalSecs int = 30

	// MinMetricsPollIntervalSecs is the minimum allowed poll interval to prevent excessive controller load.
	MinMetricsPollIntervalSecs int = 5

	// MaxMetricsPollIntervalSecs is the maximum allowed poll interval to keep tracking responsive.
	MaxMetricsPollIntervalSecs int = 300

	// PreStopBufferSecs is added to (2 × poll interval) for preStop hook duration.
	// This ensures at least 2 poll opportunities after training completion.
	PreStopBufferSecs int = 10

	// TerminationGraceBufferSecs is added to preStop duration for pod termination grace period.
	// This allows time for graceful process shutdown after preStop hook completes.
	TerminationGraceBufferSecs int = 30

	// NetworkPolicy constants for metrics endpoint security

	// DefaultControllerNamespace is the fallback when SA namespace file is unavailable.
	DefaultControllerNamespace string = "opendatahub"

	// ControllerPodLabelName is the label key used to identify the controller pod.
	// NetworkPolicy uses this to allow controller access to training pod metrics.
	ControllerPodLabelName string = "app.kubernetes.io/name"

	// ControllerPodLabelNameValue is the expected value for the controller name label.
	// Must match the label applied to controller pods in deployment manifests.
	// RHOAI: Set via kustomization.yaml labels overlay.
	ControllerPodLabelNameValue string = "trainer"

	// ControllerPodLabelComponent is the label key for component identification.
	ControllerPodLabelComponent string = "app.kubernetes.io/component"

	// ControllerPodLabelComponentValue is the expected value for the controller component label.
	// RHOAI uses "controller" (set via kustomization.yaml).
	// Upstream Kubeflow uses "manager" (set in base/manager/manager.yaml).
	// This value must match your deployment's controller pod labels.
	ControllerPodLabelComponentValue string = "controller"
)
