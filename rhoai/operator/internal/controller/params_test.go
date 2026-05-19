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

func TestReadParams(t *testing.T) {
	g := NewWithT(t)

	dir := t.TempDir()
	paramsFile := filepath.Join(dir, "params.env")
	g.Expect(os.WriteFile(paramsFile, []byte("key1=value1\nkey2=value2\n# comment\n\nkey3=value3\n"), 0o644)).To(Succeed())

	params, err := readParams(paramsFile)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params).To(HaveLen(3))
	g.Expect(params["key1"]).To(Equal("value1"))
	g.Expect(params["key2"]).To(Equal("value2"))
	g.Expect(params["key3"]).To(Equal("value3"))
}

func TestWriteParams(t *testing.T) {
	g := NewWithT(t)

	dir := t.TempDir()
	paramsFile := filepath.Join(dir, "params.env")

	params := map[string]string{"a": "1", "b": "2"}
	g.Expect(writeParams(paramsFile, params)).To(Succeed())

	readBack, err := readParams(paramsFile)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(readBack).To(Equal(params))
}

func TestResolveImageParamsOverridesFromEnv(t *testing.T) {
	g := NewWithT(t)

	dir := t.TempDir()
	overlayDir := filepath.Join(dir, defaultOverlay)
	g.Expect(os.MkdirAll(overlayDir, 0o755)).To(Succeed())

	paramsFile := filepath.Join(overlayDir, "params.env")
	g.Expect(os.WriteFile(paramsFile, []byte(imageParamControllerImage+"=default-image\n"), 0o644)).To(Succeed())

	t.Setenv("RELATED_IMAGE_ODH_TRAINER_IMAGE", "override-image")

	g.Expect(resolveImageParams(dir)).To(Succeed())

	params, err := readParams(paramsFile)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params[imageParamControllerImage]).To(Equal("override-image"))
}

func TestResolveImageParamsMissingParamsEnv(t *testing.T) {
	g := NewWithT(t)

	g.Expect(resolveImageParams(t.TempDir())).To(Succeed())
}
