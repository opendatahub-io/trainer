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
	"fmt"
	"io"
	"os"
	"path/filepath"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/render/kustomize"
)

const defaultOverlay = "rhoai"

func renderOverlay(path, namespace string) ([]unstructured.Unstructured, error) {
	opts := []kustomize.RenderOptsFn{
		kustomize.WithLabel(labels.PlatformPartOf, trainerPartOf),
	}
	if namespace != "" {
		opts = append(opts, kustomize.WithNamespace(namespace))
	}

	rendered, err := kustomize.Render(path, nil, opts...)
	if err != nil {
		return nil, fmt.Errorf("rendering kustomize overlay: %w", err)
	}

	return rendered, nil
}

func filterConfigMaps(items []unstructured.Unstructured) []unstructured.Unstructured {
	filtered := make([]unstructured.Unstructured, 0, len(items))
	for _, item := range items {
		if item.GetKind() != "ConfigMap" {
			filtered = append(filtered, item)
		}
	}
	return filtered
}

func ensureWorkDir(templatePath, workPath string) error {
	entries, err := os.ReadDir(workPath)
	if err == nil && len(entries) > 0 {
		return nil
	}

	return copyDir(templatePath, workPath)
}

func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		dstPath := filepath.Join(dst, relPath)

		if info.IsDir() {
			return os.MkdirAll(dstPath, info.Mode())
		}

		return copyFile(path, dstPath)
	})
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}

	if _, err = io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}

	return out.Close()
}
