/*
Copyright 2026.

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
	"os"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	"k8s.io/utils/ptr"
)

const (
	frameworkLabelKey = "trainer.kubeflow.org/framework"
	ancestorStepLabel = "trainer.kubeflow.org/trainjob-ancestor-step"

	frameworkTorch       = "torch"
	frameworkTrainingHub = "training-hub"

	envImageCUDA128Torch29 = "RELATED_IMAGE_ODH_TRAINING_CUDA128_TORCH29_PY312_IMAGE"
	envImageROCm64Torch29  = "RELATED_IMAGE_ODH_TRAINING_ROCM64_TORCH29_PY312_IMAGE"
	envImageTH06CUDA130    = "RELATED_IMAGE_ODH_TH06_CUDA130_TORCH210_PY312_IMAGE"
	envImageTH06ROCm64     = "RELATED_IMAGE_ODH_TH06_ROCM64_TORCH291_PY312_IMAGE"
	envImageTH06CPU        = "RELATED_IMAGE_ODH_TH06_CPU_TORCH210_PY312_IMAGE"

	defaultImageCUDA128Torch29 = "quay.io/opendatahub/odh-training-cuda128-torch29-py312@sha256:0be52d5775e95026c3899a208d9fbecb59489d48763664e842b92e66d3c112c8"
	defaultImageROCm64Torch29  = "quay.io/opendatahub/odh-training-rocm64-torch29-py312@sha256:80878d0d51fa6bc8957f669e7f3facac13669562d393a6bfc45ca8dff277c2fa"
	defaultImageTH06CUDA130    = "quay.io/opendatahub/odh-th06-cuda130-torch210-py312:odh-3.5-ea1"
	defaultImageTH06ROCm64     = "quay.io/opendatahub/odh-th06-rocm64-torch291-py312:odh-3.5-ea1"
	defaultImageTH06CPU        = "quay.io/opendatahub/odh-th06-cpu-torch210-py312:odh-3.5-ea1"
)

type runtimeConfig struct {
	Name         string
	Framework    string
	EnvVar       string
	DefaultImage string
}

var trainingRuntimes = []runtimeConfig{
	{Name: "torch-distributed", Framework: frameworkTorch, EnvVar: envImageTH06CUDA130, DefaultImage: defaultImageTH06CUDA130},
	{Name: "torch-distributed-rocm", Framework: frameworkTorch, EnvVar: envImageTH06ROCm64, DefaultImage: defaultImageTH06ROCm64},
	{Name: "torch-distributed-cpu", Framework: frameworkTorch, EnvVar: envImageTH06CPU, DefaultImage: defaultImageTH06CPU},
	{Name: "torch-distributed-cuda128-torch29-py312", Framework: frameworkTorch, EnvVar: envImageCUDA128Torch29, DefaultImage: defaultImageCUDA128Torch29},
	{Name: "torch-distributed-cuda130-torch210-py312", Framework: frameworkTorch, EnvVar: envImageTH06CUDA130, DefaultImage: defaultImageTH06CUDA130},
	{Name: "torch-distributed-rocm64-torch29-py312", Framework: frameworkTorch, EnvVar: envImageROCm64Torch29, DefaultImage: defaultImageROCm64Torch29},
	{Name: "torch-distributed-rocm64-torch291-py312", Framework: frameworkTorch, EnvVar: envImageTH06ROCm64, DefaultImage: defaultImageTH06ROCm64},
	{Name: "torch-distributed-cpu-torch210-py312", Framework: frameworkTorch, EnvVar: envImageTH06CPU, DefaultImage: defaultImageTH06CPU},

	{Name: "training-hub", Framework: frameworkTrainingHub, EnvVar: envImageTH06CUDA130, DefaultImage: defaultImageTH06CUDA130},
	{Name: "training-hub-rocm", Framework: frameworkTrainingHub, EnvVar: envImageTH06ROCm64, DefaultImage: defaultImageTH06ROCm64},
	{Name: "training-hub-cpu", Framework: frameworkTrainingHub, EnvVar: envImageTH06CPU, DefaultImage: defaultImageTH06CPU},
	{Name: "training-hub-th05-cuda128-torch29-py312", Framework: frameworkTrainingHub, EnvVar: envImageCUDA128Torch29, DefaultImage: defaultImageCUDA128Torch29},
	{Name: "training-hub-th06-cuda130-torch210-py312", Framework: frameworkTrainingHub, EnvVar: envImageTH06CUDA130, DefaultImage: defaultImageTH06CUDA130},
	{Name: "training-hub-th06-rocm64-torch291-py312", Framework: frameworkTrainingHub, EnvVar: envImageTH06ROCm64, DefaultImage: defaultImageTH06ROCm64},
	{Name: "training-hub-th06-cpu-torch210-py312", Framework: frameworkTrainingHub, EnvVar: envImageTH06CPU, DefaultImage: defaultImageTH06CPU},
}

func buildClusterTrainingRuntimes() []*trainerv1alpha1.ClusterTrainingRuntime {
	runtimes := make([]*trainerv1alpha1.ClusterTrainingRuntime, 0, len(trainingRuntimes))

	for _, cfg := range trainingRuntimes {
		image := os.Getenv(cfg.EnvVar)
		if image == "" {
			image = cfg.DefaultImage
		}

		ctr := &trainerv1alpha1.ClusterTrainingRuntime{
			TypeMeta: metav1.TypeMeta{
				APIVersion: trainerv1alpha1.SchemeGroupVersion.String(),
				Kind:       trainerv1alpha1.ClusterTrainingRuntimeKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: cfg.Name,
				Labels: map[string]string{
					frameworkLabelKey:     cfg.Framework,
					labels.PlatformPartOf: trainerPartOf,
				},
			},
			Spec: trainerv1alpha1.TrainingRuntimeSpec{
				MLPolicy: &trainerv1alpha1.MLPolicy{
					NumNodes: ptr.To[int32](1),
				},
				Template: trainerv1alpha1.JobSetTemplateSpec{
					Spec: jobsetv1alpha2.JobSetSpec{
						ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
							{
								Name: "node",
								Template: batchv1.JobTemplateSpec{
									ObjectMeta: metav1.ObjectMeta{
										Labels: map[string]string{
											ancestorStepLabel: "trainer",
										},
									},
									Spec: batchv1.JobSpec{
										Template: corev1.PodTemplateSpec{
											Spec: corev1.PodSpec{
												Containers: []corev1.Container{
													{
														Name:  "node",
														Image: image,
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		}

		runtimes = append(runtimes, ctr)
	}

	return runtimes
}
