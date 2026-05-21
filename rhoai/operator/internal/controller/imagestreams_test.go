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
	"path/filepath"
	"testing"

	. "github.com/onsi/gomega"
)

func TestResolveImageStreamParamsUsesDefaults(t *testing.T) {
	g := NewWithT(t)

	dir := createTestImageStreamManifests(t)

	g.Expect(resolveImageStreamParams(dir)).To(Succeed())

	params, err := readParams(filepath.Join(dir, "params.env"))
	g.Expect(err).NotTo(HaveOccurred())

	g.Expect(params["odh-training-universal-workbench-image-cuda-3-4"]).To(Equal("quay.io/test/cuda:3.4"))
	g.Expect(params["odh-training-universal-workbench-image-rocm-3-4"]).To(Equal("quay.io/test/rocm:3.4"))
	g.Expect(params["odh-training-universal-workbench-image-cpu-3-4"]).To(Equal("quay.io/test/cpu:3.4"))
	g.Expect(params["odh-training-universal-workbench-image-cuda-3-5"]).To(Equal("quay.io/test/cuda:3.5"))
	g.Expect(params["odh-training-universal-workbench-image-rocm-3-5"]).To(Equal("quay.io/test/rocm:3.5"))
	g.Expect(params["odh-training-universal-workbench-image-cpu-3-5"]).To(Equal("quay.io/test/cpu:3.5"))
}

func TestResolveImageStreamParamsEnvVarOverride(t *testing.T) {
	g := NewWithT(t)

	t.Setenv("RELATED_IMAGE_ODH_TRAINING_UNIVERSAL_WORKBENCH_IMAGE_CUDA", "quay.io/custom/cuda:override")
	t.Setenv("RELATED_IMAGE_ODH_TRAINING_UNIVERSAL_WORKBENCH_IMAGE_CPU_3_5", "quay.io/custom/cpu:3.5-override")

	dir := createTestImageStreamManifests(t)

	g.Expect(resolveImageStreamParams(dir)).To(Succeed())

	params, err := readParams(filepath.Join(dir, "params.env"))
	g.Expect(err).NotTo(HaveOccurred())

	g.Expect(params["odh-training-universal-workbench-image-cuda-3-4"]).To(Equal("quay.io/custom/cuda:override"))
	g.Expect(params["odh-training-universal-workbench-image-rocm-3-4"]).To(Equal("quay.io/test/rocm:3.4"))
	g.Expect(params["odh-training-universal-workbench-image-cpu-3-5"]).To(Equal("quay.io/custom/cpu:3.5-override"))
	g.Expect(params["odh-training-universal-workbench-image-cpu-3-4"]).To(Equal("quay.io/test/cpu:3.4"))
}

func TestResolveImageStreamParamsMissingFile(t *testing.T) {
	g := NewWithT(t)

	dir := t.TempDir()

	g.Expect(resolveImageStreamParams(dir)).To(Succeed())
}

func createTestImageStreamManifests(t *testing.T) string {
	t.Helper()

	dir := t.TempDir()

	paramsEnv := `odh-training-universal-workbench-image-cuda-3-4=quay.io/test/cuda:3.4
odh-training-universal-workbench-image-rocm-3-4=quay.io/test/rocm:3.4
odh-training-universal-workbench-image-cpu-3-4=quay.io/test/cpu:3.4
odh-training-universal-workbench-image-cuda-3-5=quay.io/test/cuda:3.5
odh-training-universal-workbench-image-rocm-3-5=quay.io/test/rocm:3.5
odh-training-universal-workbench-image-cpu-3-5=quay.io/test/cpu:3.5
`
	err := os.WriteFile(filepath.Join(dir, "params.env"), []byte(paramsEnv), 0o644)
	if err != nil {
		t.Fatalf("failed to write test params.env: %v", err)
	}

	return dir
}
