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
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/render/kustomize"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/resources"
)

const defaultOverlay = "rhoai"

func renderManifests(manifestsPath string) ([]unstructured.Unstructured, error) {
	overlayPath := filepath.Join(manifestsPath, defaultOverlay)
	paramsPath := filepath.Join(overlayPath, "params.env")
	if _, err := os.Stat(paramsPath); err == nil {
		if err := applyParams(paramsPath, trainerImageParamMap); err != nil {
			return nil, fmt.Errorf("applying image params: %w", err)
		}
	}

	rendered, err := kustomize.Render(overlayPath, nil,
		kustomize.WithLabel(labels.PlatformPartOf, trainerPartOf),
	)
	if err != nil {
		return nil, fmt.Errorf("rendering kustomize overlay: %w", err)
	}

	return rendered, nil
}

const fieldOwner = client.FieldOwner("trainer-module-controller")

func applyResources(ctx context.Context, c client.Client, rendered []unstructured.Unstructured) error {
	log := logf.FromContext(ctx)

	for i := range rendered {
		res := &rendered[i]

		if err := resources.Apply(ctx, c, res, fieldOwner, client.ForceOwnership); err != nil {
			return fmt.Errorf("applying %s %s/%s: %w", res.GetKind(), res.GetNamespace(), res.GetName(), err)
		}

		log.V(1).Info("Applied resource", "kind", res.GetKind(), "name", res.GetName(), "namespace", res.GetNamespace())
	}

	return nil
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
