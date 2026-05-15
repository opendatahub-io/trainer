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
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
	// +kubebuilder:scaffold:imports
)

var (
	ctx               context.Context
	cancel            context.CancelFunc
	testEnv           *envtest.Environment
	cfg               *rest.Config
	k8sClient         client.Client
	dynamicClient     dynamic.Interface
	discoveryClient   discovery.DiscoveryInterface
	testManifestsPath string
)

func TestMain(m *testing.M) {
	logf.SetLogger(zap.New(zap.WriteTo(os.Stderr), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	var err error
	err = componentsv1alpha1.AddToScheme(scheme.Scheme)
	if err != nil {
		logf.Log.Error(err, "failed to add scheme")
		os.Exit(1)
	}

	// +kubebuilder:scaffold:scheme

	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}

	if dir := getFirstFoundEnvTestBinaryDir(); dir != "" {
		testEnv.BinaryAssetsDirectory = dir
	}

	cfg, err = testEnv.Start()
	if err != nil {
		logf.Log.Error(err, "failed to start test environment")
		os.Exit(1)
	}

	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	if err != nil {
		logf.Log.Error(err, "failed to create k8s client")
		os.Exit(1)
	}

	dynamicClient, err = dynamic.NewForConfig(cfg)
	if err != nil {
		logf.Log.Error(err, "failed to create dynamic client")
		os.Exit(1)
	}

	discoveryClient, err = discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		logf.Log.Error(err, "failed to create discovery client")
		os.Exit(1)
	}

	testManifestsPath, err = createTestManifests()
	if err != nil {
		logf.Log.Error(err, "failed to create test manifests")
		os.Exit(1)
	}

	code := m.Run()

	cancel()
	if err := testEnv.Stop(); err != nil {
		logf.Log.Error(err, "failed to stop test environment")
	}

	_ = os.RemoveAll(testManifestsPath)
	// Also clean up the work dir created by ensureWorkDir
	_ = os.RemoveAll(testManifestsPath + "-work")

	os.Exit(code)
}

// getFirstFoundEnvTestBinaryDir locates the first binary in the specified path.
// ENVTEST-based tests depend on specific binaries, usually located in paths set by
// controller-runtime. When running tests directly (e.g., via an IDE) without using
// Makefile targets, the 'BinaryAssetsDirectory' must be explicitly configured.
//
// This function streamlines the process by finding the required binaries, similar to
// setting the 'KUBEBUILDER_ASSETS' environment variable. To ensure the binaries are
// properly set up, run 'make setup-envtest' beforehand.
func getFirstFoundEnvTestBinaryDir() string {
	basePath := filepath.Join("..", "..", "bin", "k8s")
	entries, err := os.ReadDir(basePath)
	if err != nil {
		logf.Log.Error(err, "Failed to read directory", "path", basePath)
		return ""
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return filepath.Join(basePath, entry.Name())
		}
	}
	return ""
}

func createTestManifests() (string, error) {
	dir, err := os.MkdirTemp("", "trainer-manifests-*")
	if err != nil {
		return "", err
	}

	overlayDir := filepath.Join(dir, defaultOverlay)
	if err := os.MkdirAll(overlayDir, 0o755); err != nil {
		return "", err
	}

	configMap := `apiVersion: v1
kind: ConfigMap
metadata:
  name: trainer-test-config
  namespace: default
`
	if err := os.WriteFile(filepath.Join(overlayDir, "configmap.yaml"), []byte(configMap), 0o644); err != nil {
		return "", err
	}

	kustomization := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- configmap.yaml
`
	if err := os.WriteFile(filepath.Join(overlayDir, "kustomization.yaml"), []byte(kustomization), 0o644); err != nil {
		return "", err
	}

	return dir, nil
}
