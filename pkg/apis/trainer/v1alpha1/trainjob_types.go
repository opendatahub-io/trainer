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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// TrainJobKind is the Kind name for the TrainJob.
	TrainJobKind string = "TrainJob"

	// TrainJobResourcesCreationFailedReason is used when TrainJob resources creation fails.
	TrainJobResourcesCreationFailedReason = "ResourcesCreationFailed"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.conditions[-1:].type`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// TrainJob represents configuration of a training job.
type TrainJob struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Specification of the desired TrainJob.
	Spec TrainJobSpec `json:"spec,omitempty"`

	// Current status of TrainJob.
	Status TrainJobStatus `json:"status,omitempty"`
}

const (
	// TrainJobSuspended means that TrainJob is suspended.
	TrainJobSuspended string = "Suspended"

	// TrainJobComplete means that the TrainJob has completed its execution.
	TrainJobComplete string = "Complete"

	// TrainJobFailed means that the actual jobs have failed its execution.
	TrainJobFailed string = "Failed"
)

const (
	// TrainJobSuspendedReason is the "Suspended" condition reason
	// when the TrainJob is suspended.
	TrainJobSuspendedReason string = "Suspended"

	// TrainJobResumedReason is the "Suspended" condition reason
	// when the TrainJob suspension is changed from True to False.
	TrainJobResumedReason string = "Resumed"

	// TrainJobRuntimeNotSupportedReason is the "Failed" condition reason
	// when the referenced TrainingRuntime is not supported.
	TrainJobRuntimeNotSupportedReason string = "TrainingRuntimeNotSupported"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +resource:path=trainjobs
// +kubebuilder:object:root=true

// TrainJobList is a collection of training jobs.
type TrainJobList struct {
	metav1.TypeMeta `json:",inline"`

	// Standard list metadata.
	metav1.ListMeta `json:"metadata,omitempty"`

	// List of TrainJobs.
	Items []TrainJob `json:"items"`
}

// TrainJobSpec represents specification of the desired TrainJob.
type TrainJobSpec struct {
	// Reference to the training runtime.
	// The field is immutable.
	// +kubebuilder:validation:XValidation:rule="self == oldSelf", message="runtimeRef is immutable"
	RuntimeRef RuntimeRef `json:"runtimeRef"`

	// Configuration of the initializer.
	Initializer *Initializer `json:"initializer,omitempty"`

	// Configuration of the trainer.
	Trainer *Trainer `json:"trainer,omitempty"`

	// Labels to apply for the derivative JobSet and Jobs.
	// They will be merged with the TrainingRuntime values.
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations to apply for the derivative JobSet and Jobs.
	// They will be merged with the TrainingRuntime values.
	Annotations map[string]string `json:"annotations,omitempty"`

	// Custom overrides for the training runtime.
	// +listType=atomic
	PodSpecOverrides []PodSpecOverride `json:"podSpecOverrides,omitempty"`

	// Whether the controller should suspend the running TrainJob.
	// Defaults to false.
	// +kubebuilder:default=false
	Suspend *bool `json:"suspend,omitempty"`

	// ManagedBy is used to indicate the controller or entity that manages a TrainJob.
	// The value must be either an empty, `trainer.kubeflow.org/trainjob-controller` or
	// `kueue.x-k8s.io/multikueue`. The built-in TrainJob controller reconciles TrainJob which
	// don't have this field at all or the field value is the reserved string
	// `trainer.kubeflow.org/trainjob-controller`, but delegates reconciling TrainJobs
	// with a 'kueue.x-k8s.io/multikueue' to the Kueue. The field is immutable.
	// Defaults to `trainer.kubeflow.org/trainjob-controller`
	// +kubebuilder:default="trainer.kubeflow.org/trainjob-controller"
	// +kubebuilder:validation:XValidation:rule="self in ['trainer.kubeflow.org/trainjob-controller', 'kueue.x-k8s.io/multikueue']", message="ManagedBy must be trainer.kubeflow.org/trainjob-controller or kueue.x-k8s.io/multikueue if set"
	// +kubebuilder:validation:XValidation:rule="self == oldSelf", message="ManagedBy value is immutable"
	ManagedBy *string `json:"managedBy,omitempty"`

	// Configuration for model checkpointing during training.
	Checkpointing *CheckpointingConfig `json:"checkpointing,omitempty"`
}

// RuntimeRef represents the reference to the existing training runtime.
type RuntimeRef struct {
	// Name of the runtime being referenced.
	// When namespaced-scoped TrainingRuntime is used, the TrainJob must have
	// the same namespace as the deployed runtime.
	Name string `json:"name"`

	// APIGroup of the runtime being referenced.
	// Defaults to `trainer.kubeflow.org`.
	// +kubebuilder:default="trainer.kubeflow.org"
	APIGroup *string `json:"apiGroup,omitempty"`

	// Kind of the runtime being referenced.
	// Defaults to ClusterTrainingRuntime.
	// +kubebuilder:default="ClusterTrainingRuntime"
	Kind *string `json:"kind,omitempty"`
}

// Initializer represents the desired configuration for the dataset and model initialization.
// It is used to initialize the assets (dataset and pre-trained model) and pre-process data.
type Initializer struct {
	// Configuration of the dataset initialization and pre-processing.
	Dataset *DatasetInitializer `json:"dataset,omitempty"`

	// Configuration of the pre-trained model initialization
	Model *ModelInitializer `json:"model,omitempty"`
}

// DatasetInitializer represents the desired configuration to initialize and pre-process dataset.
// The DatasetInitializer spec will override the runtime Job template
// which contains this label: `trainer.kubeflow.org/trainjob-ancestor-step: dataset-initializer`
type DatasetInitializer struct {
	// Storage uri for the dataset provider.
	StorageUri *string `json:"storageUri,omitempty"`

	// List of environment variables to set in the dataset initializer container.
	// These values will be merged with the TrainingRuntime's dataset initializer environments.
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Reference to the secret with credentials to download dataset.
	// Secret must be created in the TrainJob's namespace.
	SecretRef *corev1.LocalObjectReference `json:"secretRef,omitempty"`
}

// DatasetInitializer represents the desired configuration to initialize pre-trained model.
// The DatasetInitializer spec will override the runtime Job template
// which contains this label: `trainer.kubeflow.org/trainjob-ancestor-step: dataset-initializer`
type ModelInitializer struct {
	// Storage uri for the model provider.
	StorageUri *string `json:"storageUri,omitempty"`

	// List of environment variables to set in the model initializer container.
	// These values will be merged with the TrainingRuntime's model initializer environments.
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Reference to the secret with credentials to download model.
	// Secret must be created in the TrainJob's namespace.
	SecretRef *corev1.LocalObjectReference `json:"secretRef,omitempty"`
}

// Trainer represents the desired configuration for the training job.
// The Trainer spec will override the runtime template
// which contains this label: `trainer.kubeflow.org/trainjob-ancestor-step: trainer`
type Trainer struct {
	// Docker image for the training container.
	Image *string `json:"image,omitempty"`

	// Entrypoint commands for the training container.
	// +listType=atomic
	Command []string `json:"command,omitempty"`

	// Arguments to the entrypoint for the training container.
	// +listType=atomic
	Args []string `json:"args,omitempty"`

	// List of environment variables to set in the training container.
	// These values will be merged with the TrainingRuntime's trainer environments.
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Number of training nodes.
	// TODO (andreyvelich): Do we want to support dynamic num of nodes in TrainJob for PyTorch elastic: `--nnodes=1:4` ?
	NumNodes *int32 `json:"numNodes,omitempty"`

	// Compute resources for each training node.
	ResourcesPerNode *corev1.ResourceRequirements `json:"resourcesPerNode,omitempty"`

	// Number of processes/workers/slots on every training node.
	// For the Torch runtime: `auto`, `cpu`, `gpu`, or int value can be set.
	// For the MPI runtime only int value can be set.
	NumProcPerNode *intstr.IntOrString `json:"numProcPerNode,omitempty"`
}

// PodSpecOverride represents the custom overrides that will be applied for the TrainJob's resources.
type PodSpecOverride struct {
	// TrainJobs is the training job replicas in the training runtime template to apply the overrides.
	// +listType=atomic
	TargetJobs []PodSpecOverrideTargetJob `json:"targetJobs"`

	// Override for the service account.
	ServiceAccountName *string `json:"serviceAccountName,omitempty"`

	// Override for the node selector to place Pod on the specific node.
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Override for the Pod's tolerations.
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// Overrides for the Pod volume configurations.
	// +listType=map
	// +listMapKey=name
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// Overrides for the init container in the target job templates.
	// +listType=map
	// +listMapKey=name
	InitContainers []ContainerOverride `json:"initContainers,omitempty"`

	// Overrides for the containers in the target job templates.
	// +listType=map
	// +listMapKey=name
	Containers []ContainerOverride `json:"containers,omitempty"`

	// SchedulingGates overrides the scheduling gates of the Pods in the target job templates.
	// More info: https://kubernetes.io/docs/concepts/scheduling-eviction/pod-scheduling-readiness/
	// +listType=map
	// +listMapKey=name
	SchedulingGates []corev1.PodSchedulingGate `json:"schedulingGates,omitempty"`
}

type PodSpecOverrideTargetJob struct {
	// Name is the target training job name for which the PodSpec is overridden.
	Name string `json:"name"`
}

// ContainerOverride represents parameters that can be overridden using PodSpecOverrides.
type ContainerOverride struct {
	// Name for the container. TrainingRuntime must have this container.
	Name string `json:"name"`

	// List of environment variables to set in the container.
	// These values will be merged with the TrainingRuntime's environments.
	// These values can't be set for container with the name: `node`, `dataset-initializer`, or
	// `model-initializer`. For those containers the envs can only be set via Trainer or Initializer APIs.
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Pod volumes to mount into the container's filesystem.
	// +listType=map
	// +listMapKey=name
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`
}

// CheckpointingConfig represents the configuration for model checkpointing during training.
type CheckpointingConfig struct {
	// Whether to enable checkpointing for this TrainJob.
	// Defaults to false.
	// +kubebuilder:default=false
	Enabled *bool `json:"enabled,omitempty"`

	// Storage configuration for saving checkpoints.
	Storage CheckpointStorage `json:"storage,omitempty"`

	// Interval for saving checkpoints (e.g., "5m", "100s", "1h").
	// If not specified, checkpoints will be saved based on training framework defaults.
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|µs|ms|s|m|h))+$`
	Interval *string `json:"interval,omitempty"`

	// Maximum number of checkpoints to retain.
	// Older checkpoints will be automatically deleted.
	// Defaults to 3.
	// +kubebuilder:default=3
	// +kubebuilder:validation:Minimum=1
	MaxCheckpoints *int32 `json:"maxCheckpoints,omitempty"`

	// Whether to resume training from the latest checkpoint if available.
	// Defaults to true.
	// +kubebuilder:default=true
	ResumeFromCheckpoint *bool `json:"resumeFromCheckpoint,omitempty"`

	// Custom environment variables to pass checkpointing configuration to the training container.
	// These will be automatically generated based on the checkpointing config, but can be overridden.
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// CheckpointStorage represents storage configuration for checkpoints.
type CheckpointStorage struct {
	// Storage URI where checkpoints will be saved (e.g., "s3://bucket/path", "gs://bucket/path", "/mnt/shared").
	// For PersistentVolume storage, use a local path like "/checkpoints" or "/shared/checkpoints".
	URI string `json:"uri"`

	// Reference to the secret containing credentials for accessing the storage.
	// Secret must be created in the TrainJob's namespace.
	// Not required for PersistentVolume storage.
	SecretRef *corev1.LocalObjectReference `json:"secretRef,omitempty"`

	// Access mode for the storage.
	// +kubebuilder:validation:Enum=ReadWriteOnce;ReadWriteMany;ReadOnlyMany
	// +kubebuilder:default="ReadWriteMany"
	AccessMode *string `json:"accessMode,omitempty"`

	// PersistentVolume configuration for checkpoint storage.
	// References an existing PVC that must be created by the platform administrator.
	PersistentVolume *CheckpointPersistentVolumeConfig `json:"persistentVolume,omitempty"`
}

// CheckpointPersistentVolumeConfig represents PersistentVolume configuration for checkpoints.
type CheckpointPersistentVolumeConfig struct {
	// Name of an existing PVC to use for checkpoint storage.
	// The PVC must exist in the same namespace as the TrainJob.
	// +kubebuilder:validation:Required
	ClaimName string `json:"claimName"`

	// Mount path where the PV should be mounted in the training containers.
	// +kubebuilder:default="/checkpoints"
	MountPath *string `json:"mountPath,omitempty"`

	// SubPath within the PV to use for this TrainJob's checkpoints.
	// Useful for sharing a single PV across multiple TrainJobs.
	SubPath *string `json:"subPath,omitempty"`
}

// TrainJobStatus represents the current status of TrainJob.
type TrainJobStatus struct {
	// Conditions for the TrainJob.
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +patchStrategy=merge
	// +patchMergeKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// JobsStatus tracks the child Jobs in TrainJob.
	// +listType=map
	// +listMapKey=name
	JobsStatus []JobStatus `json:"jobsStatus,omitempty"`

	// TrainingProgress tracks the progress of the training job.
	TrainingProgress *TrainingProgress `json:"trainingProgress,omitempty"`
}

type JobStatus struct {
	// Name of the child Job.
	Name string `json:"name"`

	// Ready is the number of child Jobs where the number of ready pods and completed pods
	// is greater than or equal to the total expected pod count for the child Job.
	Ready int32 `json:"ready"`

	// Succeeded is the number of successfully completed child Jobs.
	Succeeded int32 `json:"succeeded"`

	// Failed is the number of failed child Jobs.
	Failed int32 `json:"failed"`

	// Active is the number of child Jobs with at least 1 pod in a running or pending state
	// which are not marked for deletion.
	Active int32 `json:"active"`

	// Suspended is the number of child Jobs which are in a suspended state.
	Suspended int32 `json:"suspended"`
}

// TrainingProgress represents the current progress of the training job.
type TrainingProgress struct {
	// Current epoch number (if applicable).
	Epoch *int32 `json:"epoch,omitempty"`

	// Total number of epochs (if known).
	TotalEpochs *int32 `json:"totalEpochs,omitempty"`

	// Current step/iteration number.
	Step *int64 `json:"step,omitempty"`

	// Total number of steps (if known).
	TotalSteps *int64 `json:"totalSteps,omitempty"`

	// Training loss value (if available).
	// +kubebuilder:validation:Type=string
	Loss *string `json:"loss,omitempty"`

	// Training accuracy (if available).
	// +kubebuilder:validation:Type=string
	Accuracy *string `json:"accuracy,omitempty"`

	// Validation loss (if available).
	// +kubebuilder:validation:Type=string
	ValidationLoss *string `json:"validationLoss,omitempty"`

	// Validation accuracy (if available).
	// +kubebuilder:validation:Type=string
	ValidationAccuracy *string `json:"validationAccuracy,omitempty"`

	// Learning rate (if available).
	// +kubebuilder:validation:Type=string
	LearningRate *string `json:"learningRate,omitempty"`

	// Percentage of training completion (0-100).
	// +kubebuilder:validation:Type=string
	PercentComplete *string `json:"percentComplete,omitempty"`

	// Estimated time remaining for training completion.
	EstimatedTimeRemaining *string `json:"estimatedTimeRemaining,omitempty"`

	// Last time the progress was updated.
	LastUpdateTime *metav1.Time `json:"lastUpdateTime,omitempty"`

	// Checkpointing status and information.
	Checkpointing *CheckpointingStatus `json:"checkpointing,omitempty"`
}

// CheckpointingStatus represents the current status of checkpointing.
type CheckpointingStatus struct {
	// Whether checkpointing is currently enabled.
	Enabled bool `json:"enabled"`

	// Path or URI of the latest checkpoint.
	LatestCheckpoint *string `json:"latestCheckpoint,omitempty"`

	// Timestamp when the latest checkpoint was created.
	LatestCheckpointTime *metav1.Time `json:"latestCheckpointTime,omitempty"`

	// Total number of checkpoints created.
	CheckpointsCreated int32 `json:"checkpointsCreated"`

	// Size of the latest checkpoint (in bytes).
	LatestCheckpointSize *int64 `json:"latestCheckpointSize,omitempty"`

	// List of available checkpoints.
	// +listType=atomic
	AvailableCheckpoints []CheckpointInfo `json:"availableCheckpoints,omitempty"`

	// Any error encountered during checkpointing.
	Error *string `json:"error,omitempty"`
}

// CheckpointInfo represents information about a specific checkpoint.
type CheckpointInfo struct {
	// Path or URI of the checkpoint.
	Path string `json:"path"`

	// Timestamp when the checkpoint was created.
	CreatedAt metav1.Time `json:"createdAt"`

	// Size of the checkpoint (in bytes).
	Size *int64 `json:"size,omitempty"`

	// Epoch number when the checkpoint was created.
	Epoch *int32 `json:"epoch,omitempty"`

	// Step number when the checkpoint was created.
	Step *int64 `json:"step,omitempty"`

	// Training metrics at the time of checkpoint creation.
	Metrics map[string]string `json:"metrics,omitempty"`
}

func init() {
	SchemeBuilder.Register(&TrainJob{}, &TrainJobList{})
}
