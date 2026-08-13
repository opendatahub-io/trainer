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

const (
	imageParamControllerImage = "odh-kubeflow-trainer-controller-image"
	paramOperatorNamespace    = "operator-namespace"
)

var trainerImageParamMap = map[string]string{
	imageParamControllerImage: "RELATED_IMAGE_ODH_TRAINER_IMAGE",
}

var imageStreamParamMap = map[string]string{
	"odh-training-universal-workbench-image-cuda-3-5": "RELATED_IMAGE_ODH_TH_TORCH_CUDA_PY312_IMAGE",
	"odh-training-universal-workbench-image-rocm-3-5": "RELATED_IMAGE_ODH_TH_TORCH_ROCM_PY312_IMAGE",
	"odh-training-universal-workbench-image-cpu-3-5":  "RELATED_IMAGE_ODH_TH_TORCH_CPU_PY312_IMAGE",
}

var runtimesParamMap = map[string]string{
	"odh-th-torch-cuda-py312-image":       "RELATED_IMAGE_ODH_TH_TORCH_CUDA_PY312_IMAGE",
	"odh-th-torch-rocm-py312-image":       "RELATED_IMAGE_ODH_TH_TORCH_ROCM_PY312_IMAGE",
	"odh-th-torch-cpu-py312-image":        "RELATED_IMAGE_ODH_TH_TORCH_CPU_PY312_IMAGE",
	"odh-speculator-model-opt-cuda-image": "RELATED_IMAGE_RHAII_MODEL_OPT_CUDA_IMAGE",
	"odh-vllm-cuda-image":                 "RELATED_IMAGE_RHAII_VLLM_CUDA_IMAGE",
}

// applyStaticParams sets params.env keys directly to the given values,
// without looking up environment variables. Used to inject runtime values
// (e.g. operator-namespace) that are derived from the CR rather than images.
func applyStaticParams(dir string, values map[string]string) error {
	paramsPath := filepath.Join(dir, "params.env")

	if _, err := os.Stat(paramsPath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	params, err := readParams(paramsPath)
	if err != nil {
		return fmt.Errorf("reading params.env: %w", err)
	}

	for key, val := range values {
		params[key] = val
	}

	return writeParams(paramsPath, params)
}

func applyParamOverrides(dir string, paramMap map[string]string) error {
	paramsPath := filepath.Join(dir, "params.env")

	if _, err := os.Stat(paramsPath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	params, err := readParams(paramsPath)
	if err != nil {
		return fmt.Errorf("reading params.env: %w", err)
	}

	for key, envVar := range paramMap {
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
