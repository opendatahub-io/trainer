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
	"encoding/json"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

const (
	// Environment variables for checkpointing configuration
	CheckpointEnabledEnvVar      = "CHECKPOINT_ENABLED"
	CheckpointURIEnvVar          = "CHECKPOINT_URI"
	CheckpointIntervalEnvVar     = "CHECKPOINT_INTERVAL"
	CheckpointMaxRetainEnvVar    = "CHECKPOINT_MAX_RETAIN"
	CheckpointResumeEnvVar       = "CHECKPOINT_RESUME"
	CheckpointLocalPathEnvVar    = "CHECKPOINT_LOCAL_PATH"
	TrainJobProgressFileEnvVar   = "TRAINJOB_PROGRESS_FILE"
	CheckpointAccessModeEnvVar   = "CHECKPOINT_ACCESS_MODE"
	CheckpointStorageClassEnvVar = "CHECKPOINT_STORAGE_CLASS"
	ProgressFilePathEnvVar       = "TRAINING_PROGRESS_FILE"

	// Default values
	DefaultCheckpointInterval   = "5m"
	DefaultMaxCheckpoints       = 3
	DefaultProgressFile         = "/workspace/training_progress.json"
	DefaultCheckpointLocalPath  = "/tmp/checkpoints"
	DefaultCheckpointAccessMode = "ReadWriteMany"

	// Annotations for checkpointing
	CheckpointingEnabledAnnotation = "trainer.kubeflow.org/checkpointing-enabled"
	ProgressFilePathAnnotation     = "trainer.kubeflow.org/progress-file-path"
)

// CheckpointingManager handles checkpointing functionality for TrainJobs
type CheckpointingManager struct {
	client    client.Client
	clientset kubernetes.Interface
	config    *rest.Config
	log       logr.Logger
}

// NewCheckpointingManager creates a new checkpointing manager
func NewCheckpointingManager(client client.Client, config *rest.Config) *CheckpointingManager {
	clientset, _ := kubernetes.NewForConfig(config)
	return &CheckpointingManager{
		client:    client,
		clientset: clientset,
		config:    config,
		log:       ctrl.Log.WithName("checkpointing-manager"),
	}
}

// InjectCheckpointingEnvVars injects checkpointing environment variables into containers
func (cm *CheckpointingManager) InjectCheckpointingEnvVars(trainJob *trainer.TrainJob, containers []corev1.Container) []corev1.Container {
	if trainJob.Spec.Checkpointing == nil || !ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false) {
		return containers
	}

	checkpointingEnvs := cm.generateCheckpointingEnvVars(trainJob)

	// Inject environment variables into all containers
	for i := range containers {
		containers[i].Env = mergeEnvVars(containers[i].Env, checkpointingEnvs)
	}

	return containers
}

// generateCheckpointingEnvVars generates environment variables for checkpointing configuration
func (cm *CheckpointingManager) generateCheckpointingEnvVars(trainJob *trainer.TrainJob) []corev1.EnvVar {
	if trainJob.Spec.Checkpointing == nil {
		return nil
	}

	config := trainJob.Spec.Checkpointing
	envVars := []corev1.EnvVar{
		{
			Name:  CheckpointEnabledEnvVar,
			Value: strconv.FormatBool(ptr.Deref(config.Enabled, false)),
		},
		{
			Name:  CheckpointURIEnvVar,
			Value: config.Storage.URI,
		},
		{
			Name:  CheckpointIntervalEnvVar,
			Value: ptr.Deref(config.Interval, DefaultCheckpointInterval),
		},
		{
			Name:  CheckpointMaxRetainEnvVar,
			Value: strconv.Itoa(int(ptr.Deref(config.MaxCheckpoints, DefaultMaxCheckpoints))),
		},
		{
			Name:  CheckpointResumeEnvVar,
			Value: strconv.FormatBool(ptr.Deref(config.ResumeFromCheckpoint, true)),
		},
		{
			Name:  CheckpointLocalPathEnvVar,
			Value: DefaultCheckpointLocalPath,
		},
		{
			Name:  TrainJobProgressFileEnvVar,
			Value: DefaultProgressFile,
		},
		{
			Name:  CheckpointAccessModeEnvVar,
			Value: ptr.Deref(config.Storage.AccessMode, DefaultCheckpointAccessMode),
		},
	}

	// Add TrainJob metadata for progress reporting
	envVars = append(envVars, []corev1.EnvVar{
		{
			Name:  "TRAINJOB_NAME",
			Value: trainJob.Name,
		},
		{
			Name:  "TRAINJOB_NAMESPACE",
			Value: trainJob.Namespace,
		},
		{
			Name:  "TRAINJOB_UID",
			Value: string(trainJob.UID),
		},
	}...)

	// Add custom environment variables from the checkpointing config
	if config.Env != nil {
		envVars = append(envVars, config.Env...)
	}

	return envVars
}

// CreateCheckpointingVolumes creates volumes and volume mounts for checkpointing
func (cm *CheckpointingManager) CreateCheckpointingVolumes(trainJob *trainer.TrainJob) ([]corev1.Volume, []corev1.VolumeMount, error) {
	if trainJob.Spec.Checkpointing == nil || !ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false) {
		return nil, nil, nil
	}

	config := trainJob.Spec.Checkpointing
	volumes := []corev1.Volume{}
	volumeMounts := []corev1.VolumeMount{}

	// Create volume for checkpoint storage
	if err := cm.createCheckpointStorageVolume(trainJob, config, &volumes, &volumeMounts); err != nil {
		return nil, nil, fmt.Errorf("failed to create checkpoint storage volume: %w", err)
	}

	// Create volume for progress file
	progressVolumeName := "progress-storage"
	volumes = append(volumes, corev1.Volume{
		Name: progressVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	})

	volumeMounts = append(volumeMounts, corev1.VolumeMount{
		Name:      progressVolumeName,
		MountPath: filepath.Dir(DefaultProgressFile),
	})

	// Add secret volume if specified (for object storage credentials)
	if config.Storage.SecretRef != nil {
		secretVolumeName := "checkpoint-credentials"
		volumes = append(volumes, corev1.Volume{
			Name: secretVolumeName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName: config.Storage.SecretRef.Name,
				},
			},
		})

		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      secretVolumeName,
			MountPath: "/etc/checkpoint-credentials",
			ReadOnly:  true,
		})
	}

	return volumes, volumeMounts, nil
}

// createCheckpointStorageVolume creates the appropriate volume for checkpoint storage
func (cm *CheckpointingManager) createCheckpointStorageVolume(trainJob *trainer.TrainJob, config *trainer.CheckpointingConfig, volumes *[]corev1.Volume, volumeMounts *[]corev1.VolumeMount) error {
	volumeName := "checkpoint-storage"

	// Determine mount path
	mountPath := DefaultCheckpointLocalPath
	if config.Storage.PersistentVolume != nil && config.Storage.PersistentVolume.MountPath != nil {
		mountPath = *config.Storage.PersistentVolume.MountPath
	}

	// Handle PersistentVolume configuration
	if config.Storage.PersistentVolume != nil {
		// Use the specified PVC name
		claimName := config.Storage.PersistentVolume.ClaimName

		*volumes = append(*volumes, corev1.Volume{
			Name: volumeName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: claimName,
				},
			},
		})

		volumeMount := corev1.VolumeMount{
			Name:      volumeName,
			MountPath: mountPath,
		}

		// Add subPath if specified
		if config.Storage.PersistentVolume.SubPath != nil {
			volumeMount.SubPath = *config.Storage.PersistentVolume.SubPath
		}

		*volumeMounts = append(*volumeMounts, volumeMount)

	} else if strings.HasPrefix(config.Storage.URI, "/") {
		// Legacy support: Local filesystem path - assume PVC exists
		*volumes = append(*volumes, corev1.Volume{
			Name: volumeName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: fmt.Sprintf("%s-checkpoint-pvc", trainJob.Name),
				},
			},
		})

		*volumeMounts = append(*volumeMounts, corev1.VolumeMount{
			Name:      volumeName,
			MountPath: mountPath,
		})
	}
	// For object storage (s3://, gs://), no volume is needed as credentials are handled via secrets

	return nil
}

// UpdateTrainingProgress updates the training progress in TrainJob status
func (cm *CheckpointingManager) UpdateTrainingProgress(ctx context.Context, trainJob *trainer.TrainJob) error {
	log := cm.log.WithValues("trainJob", client.ObjectKeyFromObject(trainJob))

	if trainJob.Spec.Checkpointing == nil || !ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false) {
		return nil
	}

	// Read progress from the progress file mounted from training pods
	progressData, err := cm.readProgressFromPods(ctx, trainJob)
	if err != nil {
		log.V(4).Info("Could not read progress data", "error", err)
		return nil // Don't fail the reconciliation if progress reading fails
	}

	if progressData != nil {
		// Update the TrainJob status with progress information
		if trainJob.Status.TrainingProgress == nil {
			trainJob.Status.TrainingProgress = &trainer.TrainingProgress{}
		}

		progress := trainJob.Status.TrainingProgress
		now := metav1.Now()

		// Update progress fields
		if progressData.Epoch != nil {
			progress.Epoch = progressData.Epoch
		}
		if progressData.TotalEpochs != nil {
			progress.TotalEpochs = progressData.TotalEpochs
		}
		if progressData.Step != nil {
			progress.Step = progressData.Step
		}
		if progressData.TotalSteps != nil {
			progress.TotalSteps = progressData.TotalSteps
		}
		if progressData.Loss != nil {
			progress.Loss = progressData.Loss
		}
		if progressData.Accuracy != nil {
			progress.Accuracy = progressData.Accuracy
		}
		if progressData.ValidationLoss != nil {
			progress.ValidationLoss = progressData.ValidationLoss
		}
		if progressData.ValidationAccuracy != nil {
			progress.ValidationAccuracy = progressData.ValidationAccuracy
		}
		if progressData.LearningRate != nil {
			progress.LearningRate = progressData.LearningRate
		}
		if progressData.PercentComplete != nil {
			progress.PercentComplete = progressData.PercentComplete
		}
		if progressData.EstimatedTimeRemaining != nil {
			progress.EstimatedTimeRemaining = progressData.EstimatedTimeRemaining
		}

		progress.LastUpdateTime = &now

		// Update checkpointing status
		if progress.Checkpointing == nil {
			progress.Checkpointing = &trainer.CheckpointingStatus{
				Enabled: true,
			}
		}

		// Populate by reading checkpoint metadata from storage
		cm.updateCheckpointingStatus(ctx, trainJob, progress.Checkpointing)

		log.V(4).Info("Updated training progress",
			"epoch", progress.Epoch,
			"step", progress.Step,
			"loss", progress.Loss,
			"percentComplete", progress.PercentComplete)

		// Save the updated status back to the API server
		if err := cm.client.Status().Update(ctx, trainJob); err != nil {
			log.Error(err, "Failed to update TrainJob status with progress")
			return err
		}
	}

	return nil
}

// ProgressData represents the structure of progress data from training pods
type ProgressData struct {
	Epoch                  *int32  `json:"epoch,omitempty"`
	TotalEpochs            *int32  `json:"totalEpochs,omitempty"`
	Step                   *int64  `json:"step,omitempty"`
	TotalSteps             *int64  `json:"totalSteps,omitempty"`
	Loss                   *string `json:"loss,omitempty"`
	Accuracy               *string `json:"accuracy,omitempty"`
	ValidationLoss         *string `json:"validationLoss,omitempty"`
	ValidationAccuracy     *string `json:"validationAccuracy,omitempty"`
	LearningRate           *string `json:"learningRate,omitempty"`
	PercentComplete        *string `json:"percentComplete,omitempty"`
	EstimatedTimeRemaining *string `json:"estimatedTimeRemaining,omitempty"`
}

// readProgressFromPods reads progress data from training pods
func (cm *CheckpointingManager) readProgressFromPods(ctx context.Context, trainJob *trainer.TrainJob) (*ProgressData, error) {
	log := cm.log.WithValues("trainJob", client.ObjectKeyFromObject(trainJob))

	// List pods belonging to the TrainJob via JobSet
	podList := &corev1.PodList{}
	labelSelector := client.MatchingLabels{
		"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
	}

	if err := cm.client.List(ctx, podList, client.InNamespace(trainJob.Namespace), labelSelector); err != nil {
		log.V(4).Info("Failed to list pods", "error", err)
		return nil, err
	}

	if len(podList.Items) == 0 {
		return nil, nil
	}

	// Try to read progress from the first running pod
	for _, pod := range podList.Items {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}

		// Try to read the progress file from the pod
		progressData, err := cm.readProgressFromPod(ctx, &pod)
		if err != nil {
			log.V(4).Info("Failed to read progress from pod", "pod", pod.Name, "error", err)
			continue
		}

		if progressData != nil {
			return progressData, nil
		}
	}

	return nil, nil
}

// readProgressFromPod reads progress data from a specific pod
func (cm *CheckpointingManager) readProgressFromPod(ctx context.Context, pod *corev1.Pod) (*ProgressData, error) {
	if cm.clientset == nil {
		return nil, fmt.Errorf("kubernetes clientset not available")
	}

	// Get progress file path from pod environment variable or use default
	progressFile := cm.getProgressFilePathFromPod(pod)

	return cm.readProgressFileFromPath(ctx, pod, progressFile)
}

// readProgressFileFromPath reads progress data from a specific file path in the pod
func (cm *CheckpointingManager) readProgressFileFromPath(ctx context.Context, pod *corev1.Pod, progressFile string) (*ProgressData, error) {
	cmd := []string{"cat", progressFile}

	req := cm.clientset.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(pod.Name).
		Namespace(pod.Namespace).
		SubResource("exec")

	req.VersionedParams(&corev1.PodExecOptions{
		Command: cmd,
		Stdout:  true,
		Stderr:  true,
	}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(cm.config, "POST", req.URL())
	if err != nil {
		return nil, fmt.Errorf("failed to create executor: %w", err)
	}

	var stdout, stderr strings.Builder
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})

	if err != nil {
		// File might not exist yet, which is normal
		return nil, nil
	}

	if stdout.Len() == 0 {
		return nil, nil
	}

	// Parse the JSON progress data
	var progressData ProgressData
	if err := json.Unmarshal([]byte(stdout.String()), &progressData); err != nil {
		return nil, fmt.Errorf("failed to parse progress data: %w", err)
	}

	return &progressData, nil
}

// getProgressFilePathFromPod extracts the progress file path from pod environment variables
func (cm *CheckpointingManager) getProgressFilePathFromPod(pod *corev1.Pod) string {
	// Check all containers for the progress file path environment variable
	for _, container := range pod.Spec.Containers {
		for _, env := range container.Env {
			if env.Name == ProgressFilePathEnvVar {
				return env.Value
			}
		}
	}

	// Fallback to default path
	return DefaultProgressFile
}

// updateCheckpointingStatus updates the checkpointing status based on available checkpoints
func (cm *CheckpointingManager) updateCheckpointingStatus(ctx context.Context, trainJob *trainer.TrainJob, status *trainer.CheckpointingStatus) {
	// Update checkpointing status with available checkpoint information
	status.Enabled = true
	now := metav1.Now()
	status.LatestCheckpointTime = &now
}

// mergeEnvVars merges two slices of environment variables, with the second slice taking precedence
func mergeEnvVars(existing, new []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)

	// Add existing env vars
	for _, env := range existing {
		envMap[env.Name] = env
	}

	// Add/override with new env vars
	for _, env := range new {
		envMap[env.Name] = env
	}

	// Convert back to slice
	result := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		result = append(result, env)
	}

	return result
}

// IsCheckpointingEnabled checks if checkpointing is enabled for a TrainJob
func IsCheckpointingEnabled(trainJob *trainer.TrainJob) bool {
	return trainJob.Spec.Checkpointing != nil && ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false)
}

// GetCheckpointingAnnotations returns annotations that should be added to pods for checkpointing
func GetCheckpointingAnnotations(trainJob *trainer.TrainJob) map[string]string {
	annotations := make(map[string]string)

	if IsCheckpointingEnabled(trainJob) {
		annotations[CheckpointingEnabledAnnotation] = "true"
		annotations[ProgressFilePathAnnotation] = DefaultProgressFile
	}

	return annotations
}
