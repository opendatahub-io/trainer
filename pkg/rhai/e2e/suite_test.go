/*
Copyright 2024 The Kubeflow Authors.

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

package test

import (
	"context"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

var (
	k8sClient     client.Client
	ctx           context.Context
	testNs        *corev1.Namespace
	suiteHasFailed bool
)

func TestRHAIE2E(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)

	ginkgo.ReportAfterEach(func(report ginkgo.SpecReport) {
		if report.Failed() {
			suiteHasFailed = true
		}
	})

	ginkgo.RunSpecs(t, "RHAI Progression Tracking E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	ctx = context.Background()

	cfg := config.GetConfigOrDie()
	gomega.ExpectWithOffset(1, cfg).NotTo(gomega.BeNil())

	err := trainer.AddToScheme(scheme.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(k8sClient).NotTo(gomega.BeNil())

	testNs = &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "rhai-e2e-progression-",
		},
	}
	gomega.Expect(k8sClient.Create(ctx, testNs)).To(gomega.Succeed())
	ginkgo.GinkgoWriter.Printf("Created shared test namespace: %s\n", testNs.Name)
})

var _ = ginkgo.AfterSuite(func() {
	// Cleanup namespace only on success; keep for debugging on failure
	if testNs != nil && k8sClient != nil {
		if !suiteHasFailed {
			ginkgo.GinkgoWriter.Printf("✓ All tests passed - cleaning up: %s\n", testNs.Name)
			if err := k8sClient.Delete(ctx, testNs); err != nil {
				ginkgo.GinkgoWriter.Printf("Warning: Failed to delete namespace %s: %v\n", testNs.Name, err)
			}
		} else {
			ginkgo.GinkgoWriter.Printf("✗ Tests failed - keeping namespace for debugging: %s\n", testNs.Name)
			ginkgo.GinkgoWriter.Printf("  To cleanup: oc delete namespace %s\n", testNs.Name)
		}
	}
})
