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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"

	operatortls "github.com/opendatahub-io/trainer-operator/internal/tls"

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

const (
	// operandConfigMapPrefix matches the kubeflow-trainer-config ConfigMap
	// generated from controller_manager_config.yaml (the generator appends
	// a content hash to the name).
	operandConfigMapPrefix = "kubeflow-trainer-config"
	// operandConfigKey is the ConfigMap key holding the operand's
	// controller configuration file.
	operandConfigKey = "controller_manager_config.yaml"
	// operandDeploymentName is the operand's controller manager deployment.
	operandDeploymentName = "kubeflow-trainer-controller-manager"
	// operandTLSConfigHashAnnotation carries a hash of the rendered
	// operand configuration. The operand validates and applies its TLS
	// options at controller startup, so a configuration change must roll
	// the deployment to take effect.
	operandTLSConfigHashAnnotation = "trainer.platform.opendatahub.io/tls-config-hash"
)

// applyOperandTLSOptions injects the effective cluster TLS profile into the
// operand's controller configuration as a tls section (minVersion,
// cipherSuites, curvePreferences; translated to the operand's vocabulary),
// and annotates the operand deployment with a hash of the rendered
// configuration so configuration changes roll the operand pods.
func applyOperandTLSOptions(ctx context.Context, items []unstructured.Unstructured, profile configv1.TLSProfileSpec) error {
	opts, err := operatortls.TranslateProfile(ctx, profile)
	if err != nil {
		return fmt.Errorf("translating cluster TLS profile for the operand: %w", err)
	}

	var configHash string
	for i := range items {
		item := &items[i]
		if item.GetKind() != "ConfigMap" || !strings.HasPrefix(item.GetName(), operandConfigMapPrefix) {
			continue
		}

		raw, found, err := unstructured.NestedString(item.Object, "data", operandConfigKey)
		if err != nil {
			return fmt.Errorf("reading operand config key: %w", err)
		}
		if !found {
			continue
		}

		var cfg map[string]any
		if err := yaml.Unmarshal([]byte(raw), &cfg); err != nil {
			return fmt.Errorf("parsing operand config: %w", err)
		}
		// yaml.Unmarshal leaves cfg nil for an empty, comment-only or
		// null document; assigning to a nil map would panic in the
		// reconcile path.
		if cfg == nil {
			cfg = map[string]any{}
		}
		cfg["tls"] = opts

		out, err := yaml.Marshal(cfg)
		if err != nil {
			return fmt.Errorf("marshalling operand config: %w", err)
		}
		if err := unstructured.SetNestedField(item.Object, string(out), "data", operandConfigKey); err != nil {
			return fmt.Errorf("writing operand config: %w", err)
		}

		sum := sha256.Sum256(out)
		configHash = hex.EncodeToString(sum[:])
	}

	if configHash == "" {
		return nil
	}

	for i := range items {
		item := &items[i]
		if item.GetKind() != "Deployment" || item.GetName() != operandDeploymentName {
			continue
		}
		annotations, _, err := unstructured.NestedStringMap(item.Object, "spec", "template", "metadata", "annotations")
		if err != nil {
			return fmt.Errorf("reading operand deployment annotations: %w", err)
		}
		if annotations == nil {
			annotations = map[string]string{}
		}
		annotations[operandTLSConfigHashAnnotation] = configHash
		if err := unstructured.SetNestedStringMap(item.Object, annotations, "spec", "template", "metadata", "annotations"); err != nil {
			return fmt.Errorf("writing operand deployment annotations: %w", err)
		}
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

		if info.Mode()&os.ModeSymlink != 0 {
			return nil
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
