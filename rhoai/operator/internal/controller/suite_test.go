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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
	// +kubebuilder:scaffold:imports
)

var (
	ctx                  context.Context
	cancel               context.CancelFunc
	testEnv              *envtest.Environment
	cfg                  *rest.Config
	k8sClient            client.Client
	dynamicClient        dynamic.Interface
	discoveryClient      discovery.DiscoveryInterface
	testManifestsPath    string
	testRuntimesPath     string
	testImageStreamsPath string
	testWorkDir          string
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

	err = trainerv1alpha1.AddToScheme(scheme.Scheme)
	if err != nil {
		logf.Log.Error(err, "failed to add trainer scheme")
		os.Exit(1)
	}

	// +kubebuilder:scaffold:scheme

	trainerCRDPath := filepath.Join(
		"..", "..", "bin", "trainer-crds",
	)
	if _, err := os.Stat(trainerCRDPath); os.IsNotExist(err) {
		trainerCRDPath = findModuleCRDPath()
	}

	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases"), trainerCRDPath},
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

	testRuntimesPath, err = createTestRuntimes()
	if err != nil {
		logf.Log.Error(err, "failed to create test runtimes")
		os.Exit(1)
	}

	testImageStreamsPath, err = createTestImageStreams()
	if err != nil {
		logf.Log.Error(err, "failed to create test imagestreams")
		os.Exit(1)
	}

	testWorkDir, err = os.MkdirTemp("", "trainer-work-*")
	if err != nil {
		logf.Log.Error(err, "failed to create test work dir")
		os.Exit(1)
	}

	code := m.Run()

	cancel()
	if err := testEnv.Stop(); err != nil {
		logf.Log.Error(err, "failed to stop test environment")
	}

	_ = os.RemoveAll(testManifestsPath)
	_ = os.RemoveAll(testRuntimesPath)
	_ = os.RemoveAll(testImageStreamsPath)
	_ = os.RemoveAll(testWorkDir)

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

	paramsEnv := imageParamControllerImage + "=quay.io/test/trainer:latest\n"
	if err := os.WriteFile(filepath.Join(overlayDir, "params.env"), []byte(paramsEnv), 0o644); err != nil {
		return "", err
	}

	componentMetadata := `releases:
  - name: Kubeflow Trainer
    version: "2.1.0"
    repoUrl: https://github.com/kubeflow/trainer
`
	if err := os.WriteFile(filepath.Join(dir, "component_metadata.yaml"), []byte(componentMetadata), 0o644); err != nil {
		return "", err
	}

	return dir, nil
}

func createTestRuntimes() (string, error) {
	dir, err := os.MkdirTemp("", "trainer-runtimes-*")
	if err != nil {
		return "", err
	}

	ctr := `apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: test-runtime
  labels:
    trainer.kubeflow.org/framework: torch
spec:
  mlPolicy:
    numNodes: 1
  template:
    spec:
      replicatedJobs:
        - name: node
          template:
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: quay.io/test/cuda:latest
`
	if err := os.WriteFile(filepath.Join(dir, "test_runtime.yaml"), []byte(ctr), 0o644); err != nil {
		return "", err
	}

	kustomization := `apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- test_runtime.yaml
`
	if err := os.WriteFile(filepath.Join(dir, "kustomization.yaml"), []byte(kustomization), 0o644); err != nil {
		return "", err
	}

	paramsEnv := `odh-training-cuda128-torch29-py312-image=quay.io/test/cuda128:latest
odh-training-rocm64-torch29-py312-image=quay.io/test/rocm64:latest
odh-th06-cuda130-torch210-py312-image=quay.io/test/cuda:latest
odh-th06-rocm64-torch291-py312-image=quay.io/test/rocm:latest
odh-th06-cpu-torch210-py312-image=quay.io/test/cpu:latest
`
	if err := os.WriteFile(filepath.Join(dir, "params.env"), []byte(paramsEnv), 0o644); err != nil {
		return "", err
	}

	return dir, nil
}

func createTestImageStreams() (string, error) {
	dir, err := os.MkdirTemp("", "trainer-imagestreams-*")
	if err != nil {
		return "", err
	}

	paramsEnv := `odh-training-universal-workbench-image-cuda-3-4=quay.io/test/cuda:3.4
odh-training-universal-workbench-image-rocm-3-4=quay.io/test/rocm:3.4
odh-training-universal-workbench-image-cpu-3-4=quay.io/test/cpu:3.4
odh-training-universal-workbench-image-cuda-3-5=quay.io/test/cuda:3.5
odh-training-universal-workbench-image-rocm-3-5=quay.io/test/rocm:3.5
odh-training-universal-workbench-image-cpu-3-5=quay.io/test/cpu:3.5
`
	if err := os.WriteFile(filepath.Join(dir, "params.env"), []byte(paramsEnv), 0o644); err != nil {
		return "", err
	}

	return dir, nil
}

func findModuleCRDPath() string {
	out, err := exec.Command("go", "env", "GOPATH").Output()
	if err != nil {
		return ""
	}
	gopath := strings.TrimSpace(string(out))
	return filepath.Join(gopath, "pkg", "mod", "github.com", "kubeflow", "trainer", "v2@v2.2.0", "manifests", "base", "crds")
}
