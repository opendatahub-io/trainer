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

package jobset

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/utils/ptr"
	jobsetv1alpha2ac "sigs.k8s.io/jobset/client-go/applyconfiguration/jobset/v1alpha2"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/apply"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	"github.com/kubeflow/trainer/v2/pkg/runtime"
	jobsetplgconsts "github.com/kubeflow/trainer/v2/pkg/runtime/framework/plugins/jobset/constants"
)

type Builder struct {
	*jobsetv1alpha2ac.JobSetApplyConfiguration
}

func NewBuilder(jobSet *jobsetv1alpha2ac.JobSetApplyConfiguration) *Builder {
	return &Builder{
		JobSetApplyConfiguration: jobSet,
	}
}

// Initializer updates JobSet values for the initializer Job.
func (b *Builder) Initializer(trainJob *trainer.TrainJob) *Builder {
	for i, rJob := range b.Spec.ReplicatedJobs {
		jobMetadata := rJob.Template.ObjectMetaApplyConfiguration
		if jobMetadata == nil || jobMetadata.Labels == nil {
			continue
		}
		// Update values for the Dataset Initializer Job.
		if ancestor, ok := jobMetadata.Labels[constants.LabelTrainJobAncestor]; ok && ancestor == constants.DatasetInitializer {
			// TODO: Support multiple replicas ('.template.spec.replicatedJobs[*].replicas') for replicated Jobs.
			// REF: https://github.com/kubeflow/trainer/issues/2318
			b.Spec.ReplicatedJobs[i].Replicas = ptr.To[int32](1)
			for j, container := range rJob.Template.Spec.Template.Spec.Containers {
				// Update values for the dataset initializer container.
				if *container.Name == constants.DatasetInitializer && trainJob.Spec.Initializer != nil && trainJob.Spec.Initializer.Dataset != nil {
					env := &b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Env
					// Update the dataset initializer envs.
					if storageUri := trainJob.Spec.Initializer.Dataset.StorageUri; storageUri != nil {
						apply.UpsertEnvVar(env, *corev1ac.EnvVar().
							WithName(jobsetplgconsts.InitializerEnvStorageUri).
							WithValue(*storageUri))
					}
					apply.UpsertEnvVars(env, apply.EnvVars(trainJob.Spec.Initializer.Dataset.Env...)...)
					// Update the dataset initializer secret reference.
					if trainJob.Spec.Initializer.Dataset.SecretRef != nil {
						b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].
							WithEnvFrom(corev1ac.EnvFromSource().
								WithSecretRef(corev1ac.SecretEnvSource().
									WithName(trainJob.Spec.Initializer.Dataset.SecretRef.Name)))
					}
				}
			}
		}
		// Update values for the Model Initializer Job.
		if ancestor, ok := jobMetadata.Labels[constants.LabelTrainJobAncestor]; ok && ancestor == constants.ModelInitializer {
			// TODO: Support multiple replicas ('.template.spec.replicatedJobs[*].replicas') for replicated Jobs.
			// REF: https://github.com/kubeflow/trainer/issues/2318
			b.Spec.ReplicatedJobs[i].Replicas = ptr.To[int32](1)
			for j, container := range rJob.Template.Spec.Template.Spec.Containers {
				// Update values for the model initializer container.
				if *container.Name == constants.ModelInitializer && trainJob.Spec.Initializer != nil && trainJob.Spec.Initializer.Model != nil {
					env := &b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Env
					// Update the model initializer envs.
					if storageUri := trainJob.Spec.Initializer.Model.StorageUri; storageUri != nil {
						apply.UpsertEnvVar(env, *corev1ac.EnvVar().
							WithName(jobsetplgconsts.InitializerEnvStorageUri).
							WithValue(*storageUri))
					}
					apply.UpsertEnvVars(env, apply.EnvVars(trainJob.Spec.Initializer.Model.Env...)...)
					// Update the model initializer secret reference.
					if trainJob.Spec.Initializer.Model.SecretRef != nil {
						b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].
							WithEnvFrom(corev1ac.EnvFromSource().
								WithSecretRef(corev1ac.SecretEnvSource().
									WithName(trainJob.Spec.Initializer.Model.SecretRef.Name)))
					}
				}
			}
		}
	}
	return b
}

// Trainer updates JobSet values for the trainer Job.
func (b *Builder) Trainer(info *runtime.Info, trainJob *trainer.TrainJob) *Builder {
	for i, rJob := range b.Spec.ReplicatedJobs {
		jobMetadata := rJob.Template.ObjectMetaApplyConfiguration
		if jobMetadata == nil || jobMetadata.Labels == nil {
			continue
		}
		if ancestor, ok := jobMetadata.Labels[constants.LabelTrainJobAncestor]; ok && ancestor == constants.AncestorTrainer {
			// TODO: Support multiple replicas ('.template.spec.replicatedJobs[*].replicas') for replicated Jobs.
			// REF: https://github.com/kubeflow/trainer/issues/2318
			b.Spec.ReplicatedJobs[i].Replicas = ptr.To[int32](1)
			// Update the Parallelism and Completions values for the Trainer Job.
			b.Spec.ReplicatedJobs[i].Template.Spec.Parallelism = info.FindPodSetByAncestor(constants.AncestorTrainer).Count
			b.Spec.ReplicatedJobs[i].Template.Spec.Completions = info.FindPodSetByAncestor(constants.AncestorTrainer).Count

			// Update values for the Trainer container.
			for j, container := range rJob.Template.Spec.Template.Spec.Containers {
				if *container.Name == constants.Node {
					// Update values from the TrainJob trainer.
					if jobTrainer := trainJob.Spec.Trainer; jobTrainer != nil {
						if image := jobTrainer.Image; image != nil {
							b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Image = image
						}
						if command := jobTrainer.Command; command != nil {
							b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Command = command
						}
						if args := jobTrainer.Args; args != nil {
							b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Args = args
						}
						if resourcesPerNode := jobTrainer.ResourcesPerNode; resourcesPerNode != nil &&
							(resourcesPerNode.Limits != nil || resourcesPerNode.Requests != nil) {
							requirements := corev1ac.ResourceRequirements()
							if limits := resourcesPerNode.Limits; limits != nil {
								requirements.WithLimits(limits)
							}
							if requests := resourcesPerNode.Requests; requests != nil {
								requirements.WithRequests(requests)
							}
							b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].
								WithResources(requirements)
						}
						apply.UpsertEnvVars(
							&b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Env,
							apply.EnvVars(jobTrainer.Env...)...,
						)
					}

					// Add checkpointing environment variables if enabled
					if trainJob.Spec.Checkpointing != nil && ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false) {
						checkpointingEnvs := b.generateCheckpointingEnvVars(trainJob)
						for _, envVar := range checkpointingEnvs {
							apply.UpsertEnvVar(&b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].Env, *envVar)
						}
					}
				}
			}

			// Add checkpointing volumes and volume mounts if enabled
			if trainJob.Spec.Checkpointing != nil && ptr.Deref(trainJob.Spec.Checkpointing.Enabled, false) {
				volumes, volumeMounts := b.generateCheckpointingVolumes(trainJob)

				// Add volumes to the pod spec
				if len(volumes) > 0 {
					volumeConfigs := make([]corev1ac.VolumeApplyConfiguration, len(volumes))
					for idx, v := range volumes {
						volumeConfigs[idx] = *v
					}
					apply.UpsertVolumes(&b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Volumes, volumeConfigs...)
				}

				// Add volume mounts to the trainer container
				for j, container := range rJob.Template.Spec.Template.Spec.Containers {
					if *container.Name == constants.Node && len(volumeMounts) > 0 {
						mountConfigs := make([]corev1ac.VolumeMountApplyConfiguration, len(volumeMounts))
						for k, vm := range volumeMounts {
							mountConfigs[k] = *vm
						}
						apply.UpsertVolumeMounts(&b.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers[j].VolumeMounts, mountConfigs...)
					}
				}
			}
		}
	}
	return b
}

// TODO: Supporting merge labels would be great.

func (b *Builder) PodLabels(labels map[string]string) *Builder {
	for i := range b.Spec.ReplicatedJobs {
		b.Spec.ReplicatedJobs[i].Template.Spec.Template.WithLabels(labels)
	}
	return b
}

func (b *Builder) Suspend(suspend *bool) *Builder {
	b.Spec.Suspend = suspend
	return b
}

func (b *Builder) Build() *jobsetv1alpha2ac.JobSetApplyConfiguration {
	return b.JobSetApplyConfiguration
}

// generateCheckpointingEnvVars generates environment variables for checkpointing configuration
func (b *Builder) generateCheckpointingEnvVars(trainJob *trainer.TrainJob) []*corev1ac.EnvVarApplyConfiguration {
	if trainJob.Spec.Checkpointing == nil {
		return nil
	}

	config := trainJob.Spec.Checkpointing
	envVars := []*corev1ac.EnvVarApplyConfiguration{
		corev1ac.EnvVar().WithName("CHECKPOINT_ENABLED").WithValue("true"),
		corev1ac.EnvVar().WithName("CHECKPOINT_URI").WithValue(config.Storage.URI),
		corev1ac.EnvVar().WithName("CHECKPOINT_INTERVAL").WithValue(ptr.Deref(config.Interval, "5m")),
		corev1ac.EnvVar().WithName("CHECKPOINT_MAX_RETAIN").WithValue(strconv.Itoa(int(ptr.Deref(config.MaxCheckpoints, 3)))),
		corev1ac.EnvVar().WithName("CHECKPOINT_RESUME").WithValue(strconv.FormatBool(ptr.Deref(config.ResumeFromCheckpoint, true))),
		corev1ac.EnvVar().WithName("CHECKPOINT_LOCAL_PATH").WithValue("/tmp/checkpoints"),
		corev1ac.EnvVar().WithName("TRAINJOB_PROGRESS_FILE").WithValue("/tmp/training_progress.json"),
		corev1ac.EnvVar().WithName("CHECKPOINT_ACCESS_MODE").WithValue(ptr.Deref(config.Storage.AccessMode, "ReadWriteMany")),
		corev1ac.EnvVar().WithName("TRAINJOB_NAME").WithValue(trainJob.Name),
		corev1ac.EnvVar().WithName("TRAINJOB_NAMESPACE").WithValue(trainJob.Namespace),
		corev1ac.EnvVar().WithName("TRAINJOB_UID").WithValue(string(trainJob.UID)),
	}

	// Add custom environment variables from the checkpointing config
	for _, env := range config.Env {
		envVars = append(envVars, corev1ac.EnvVar().WithName(env.Name).WithValue(env.Value))
	}

	return envVars
}

// generateCheckpointingVolumes generates volumes and volume mounts for checkpointing
func (b *Builder) generateCheckpointingVolumes(trainJob *trainer.TrainJob) ([]*corev1ac.VolumeApplyConfiguration, []*corev1ac.VolumeMountApplyConfiguration) {
	if trainJob.Spec.Checkpointing == nil {
		return nil, nil
	}

	config := trainJob.Spec.Checkpointing
	volumes := []*corev1ac.VolumeApplyConfiguration{}
	volumeMounts := []*corev1ac.VolumeMountApplyConfiguration{}

	// Create volume for checkpoint storage
	if config.Storage.PersistentVolume != nil {
		volumeName := "checkpoint-storage"

		// Use the specified PVC name
		claimName := config.Storage.PersistentVolume.ClaimName

		// Create PVC volume
		volumes = append(volumes, corev1ac.Volume().
			WithName(volumeName).
			WithPersistentVolumeClaim(corev1ac.PersistentVolumeClaimVolumeSource().
				WithClaimName(claimName)))

		// Determine mount path
		mountPath := "/tmp/checkpoints"
		if config.Storage.PersistentVolume != nil && config.Storage.PersistentVolume.MountPath != nil {
			mountPath = *config.Storage.PersistentVolume.MountPath
		}

		// Create volume mount
		volumeMount := corev1ac.VolumeMount().
			WithName(volumeName).
			WithMountPath(mountPath)

		// Add subPath if specified
		if config.Storage.PersistentVolume != nil && config.Storage.PersistentVolume.SubPath != nil {
			volumeMount.WithSubPath(*config.Storage.PersistentVolume.SubPath)
		}

		volumeMounts = append(volumeMounts, volumeMount)
	} else if strings.HasPrefix(config.Storage.URI, "/") {
		// Legacy support: Local filesystem path - assume PVC exists
		volumeName := "checkpoint-storage"
		claimName := fmt.Sprintf("%s-checkpoint-pvc", trainJob.Name)

		// Create PVC volume
		volumes = append(volumes, corev1ac.Volume().
			WithName(volumeName).
			WithPersistentVolumeClaim(corev1ac.PersistentVolumeClaimVolumeSource().
				WithClaimName(claimName)))

		// Use default mount path
		mountPath := "/tmp/checkpoints"
		volumeMounts = append(volumeMounts, corev1ac.VolumeMount().
			WithName(volumeName).
			WithMountPath(mountPath))
	}

	// Create volume for progress file
	progressVolumeName := "progress-storage"
	volumes = append(volumes, corev1ac.Volume().
		WithName(progressVolumeName).
		WithEmptyDir(corev1ac.EmptyDirVolumeSource()))

	volumeMounts = append(volumeMounts, corev1ac.VolumeMount().
		WithName(progressVolumeName).
		WithMountPath(filepath.Dir("/tmp/training_progress.json")))

	// Add secret volume if specified (for object storage credentials)
	if config.Storage.SecretRef != nil {
		secretVolumeName := "checkpoint-credentials"
		volumes = append(volumes, corev1ac.Volume().
			WithName(secretVolumeName).
			WithSecret(corev1ac.SecretVolumeSource().
				WithSecretName(config.Storage.SecretRef.Name)))

		volumeMounts = append(volumeMounts, corev1ac.VolumeMount().
			WithName(secretVolumeName).
			WithMountPath("/etc/checkpoint-credentials").
			WithReadOnly(true))
	}

	return volumes, volumeMounts
}
