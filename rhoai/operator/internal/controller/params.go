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

// [TEMPORARY] params.env parsing — will migrate to odh-platform-utilities when available.
// See kserve-module/pkg/kservemodule/params.go for the equivalent temporary implementation.
package controller

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var trainerImageParamMap = map[string]string{
	"odh-kubeflow-trainer-controller-image":           "RELATED_IMAGE_ODH_TRAINER_IMAGE",
	"odh-training-cuda128-torch29-py312-image":        "RELATED_IMAGE_ODH_TRAINING_CUDA128_TORCH29_PY312_IMAGE",
	"odh-training-rocm64-torch29-py312-image":         "RELATED_IMAGE_ODH_TRAINING_ROCM64_TORCH29_PY312_IMAGE",
	"odh-th06-cuda130-torch210-py312-image":           "RELATED_IMAGE_ODH_TH06_CUDA130_TORCH210_PY312_IMAGE",
	"odh-th06-rocm64-torch291-py312-image":            "RELATED_IMAGE_ODH_TH06_ROCM64_TORCH291_PY312_IMAGE",
	"odh-th06-cpu-torch210-py312-image":               "RELATED_IMAGE_ODH_TH06_CPU_TORCH210_PY312_IMAGE",
	"odh-training-universal-workbench-image-cuda-3-4": "RELATED_IMAGE_ODH_TRAINING_UNIVERSAL_WORKBENCH_IMAGE_CUDA_3_4",
	"odh-training-universal-workbench-image-rocm-3-4": "RELATED_IMAGE_ODH_TRAINING_UNIVERSAL_WORKBENCH_IMAGE_ROCM_3_4",
	"odh-training-universal-workbench-image-cpu-3-4":  "RELATED_IMAGE_ODH_TRAINING_UNIVERSAL_WORKBENCH_IMAGE_CPU_3_4",
}

func applyParams(paramsPath string, imageMap map[string]string) error {
	params, err := readParams(paramsPath)
	if err != nil {
		return fmt.Errorf("reading params.env: %w", err)
	}

	for key, envVar := range imageMap {
		if val := os.Getenv(envVar); val != "" {
			params[key] = val
		}
	}

	return writeParams(paramsPath, params)
}

func readParams(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	params := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		params[key] = val
	}

	return params, scanner.Err()
}

func writeParams(path string, params map[string]string) error {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	w := bufio.NewWriter(f)
	for key, val := range params {
		if _, err := fmt.Fprintf(w, "%s=%s\n", key, val); err != nil {
			_ = f.Close()
			_ = os.Remove(tmpPath)
			return err
		}
	}

	if err := w.Flush(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return err
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}

	return os.Rename(tmpPath, filepath.Clean(path))
}
