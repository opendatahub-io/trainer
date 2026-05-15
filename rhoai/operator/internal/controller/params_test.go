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

func TestApplyParamsOverridesFromEnv(t *testing.T) {
	g := NewWithT(t)

	dir := t.TempDir()
	paramsFile := filepath.Join(dir, "params.env")
	g.Expect(os.WriteFile(paramsFile, []byte("img1=default1\nimg2=default2\n"), 0o644)).To(Succeed())

	t.Setenv("TEST_IMG1_ENV", "override1")

	imageMap := map[string]string{
		"img1": "TEST_IMG1_ENV",
		"img2": "TEST_IMG2_ENV_UNSET",
	}

	g.Expect(applyParams(paramsFile, imageMap)).To(Succeed())

	params, err := readParams(paramsFile)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(params["img1"]).To(Equal("override1"))
	g.Expect(params["img2"]).To(Equal("default2"))
}
