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
	"testing"

	. "github.com/onsi/gomega"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"
)

func TestBuildClusterTrainingRuntimes(t *testing.T) {
	g := NewWithT(t)

	ctrs := buildClusterTrainingRuntimes()
	g.Expect(ctrs).To(HaveLen(15))

	for _, ctr := range ctrs {
		g.Expect(ctr.Labels).To(HaveKey(frameworkLabelKey))
		g.Expect(ctr.Labels).To(HaveKeyWithValue(labels.PlatformPartOf, trainerPartOf))
		g.Expect(ctr.Spec.MLPolicy).NotTo(BeNil())
		g.Expect(ctr.Spec.MLPolicy.Torch).To(BeNil())
		g.Expect(*ctr.Spec.MLPolicy.NumNodes).To(Equal(int32(1)))

		container := ctr.Spec.Template.Spec.ReplicatedJobs[0].Template.Spec.Template.Spec.Containers[0]
		g.Expect(container.Name).To(Equal("node"))
		g.Expect(container.Image).NotTo(BeEmpty())
	}
}

func TestBuildClusterTrainingRuntimesUsesDefaults(t *testing.T) {
	g := NewWithT(t)

	ctrs := buildClusterTrainingRuntimes()

	byName := make(map[string]string)
	for _, ctr := range ctrs {
		byName[ctr.Name] = ctr.Spec.Template.Spec.ReplicatedJobs[0].Template.Spec.Template.Spec.Containers[0].Image
	}

	g.Expect(byName["torch-distributed"]).To(Equal(defaultImageTH06CUDA130))
	g.Expect(byName["torch-distributed-rocm"]).To(Equal(defaultImageTH06ROCm64))
	g.Expect(byName["torch-distributed-cpu"]).To(Equal(defaultImageTH06CPU))
	g.Expect(byName["training-hub"]).To(Equal(defaultImageTH06CUDA130))
	g.Expect(byName["training-hub-cpu"]).To(Equal(defaultImageTH06CPU))
}

func TestBuildClusterTrainingRuntimesEnvVarOverride(t *testing.T) {
	g := NewWithT(t)

	t.Setenv(envImageTH06CUDA130, "quay.io/custom/cuda:override")

	ctrs := buildClusterTrainingRuntimes()

	byName := make(map[string]string)
	for _, ctr := range ctrs {
		byName[ctr.Name] = ctr.Spec.Template.Spec.ReplicatedJobs[0].Template.Spec.Template.Spec.Containers[0].Image
	}

	g.Expect(byName["torch-distributed"]).To(Equal("quay.io/custom/cuda:override"))
	g.Expect(byName["training-hub"]).To(Equal("quay.io/custom/cuda:override"))
	g.Expect(byName["torch-distributed-cpu"]).To(Equal(defaultImageTH06CPU))
}

func TestBuildClusterTrainingRuntimesFrameworkLabels(t *testing.T) {
	g := NewWithT(t)

	ctrs := buildClusterTrainingRuntimes()

	for _, ctr := range ctrs {
		if ctr.Name == "training-hub" || ctr.Name == "training-hub-cpu" || ctr.Name == "training-hub-rocm" {
			g.Expect(ctr.Labels[frameworkLabelKey]).To(Equal(frameworkTrainingHub), "CTR %s", ctr.Name)
		}
		if ctr.Name == "torch-distributed" || ctr.Name == "torch-distributed-cpu" {
			g.Expect(ctr.Labels[frameworkLabelKey]).To(Equal(frameworkTorch), "CTR %s", ctr.Name)
		}
	}
}
