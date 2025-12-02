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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
	"github.com/kubeflow/trainer/v2/pkg/rhai/progression"
	testingutil "github.com/kubeflow/trainer/v2/pkg/util/testing"
)

const (
	timeout            = 5 * time.Minute
	interval           = 2 * time.Second
	consistentDuration = 8 * time.Second
	wrapperTestRuntime = "wrapper-test-runtime"

	// SDK poll interval constraints: 5-300 seconds
	minSDKPollIntervalSeconds  = 5
	maxSDKPollIntervalSeconds  = 300
	defaultPollIntervalSeconds = 30
)

// loadRuntimeFromFile loads TrainingRuntime from YAML file with optional unique name
func loadRuntimeFromFile(filePath, namespace, uniqueName string) (*trainer.TrainingRuntime, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	runtime := &trainer.TrainingRuntime{}
	if err := yaml.Unmarshal(data, runtime); err != nil {
		return nil, err
	}

	runtime.Namespace = namespace
	if uniqueName != "" {
		runtime.Name = uniqueName
	}
	return runtime, nil
}

var _ = ginkgo.Describe("RHAI Progression Tracking E2E Tests", ginkgo.Serial, func() {
	var runtime *trainer.TrainingRuntime

	ginkgo.BeforeEach(func() {
		runtimeFile := filepath.Join("..", "test", "resources", "wrapper-test-runtime.yaml")
		uniqueName := fmt.Sprintf("wrapper-test-runtime-%d", time.Now().UnixNano())
		var err error
		runtime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(k8sClient.Create(ctx, runtime)).To(gomega.Succeed())

		gomega.Eventually(func(g gomega.Gomega) {
			gotRuntime := &trainer.TrainingRuntime{}
			g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(runtime), gotRuntime)).Should(gomega.Succeed())
		}, timeout, interval).Should(gomega.Succeed())

		// Use DeferCleanup to ensure runtime is deleted after test completes
		// This prevents webhook validation errors when controller updates annotations
		ginkgo.DeferCleanup(func() {
			if runtime != nil {
				// Give controller time to finish final reconciliation before deleting runtime
				time.Sleep(2 * time.Second)
				gomega.Expect(k8sClient.Delete(ctx, runtime)).To(gomega.Succeed())
			}
		})
	})

	ginkgo.Context("When progression tracking is enabled", func() {
		ginkgo.It("should poll metrics and update trainerStatus annotation during training", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-enabled").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "5s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("4"),
							corev1.ResourceMemory: resource.MustParse("8Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with progression tracking enabled")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob pod to be running with IP")
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()), "at least one pod should exist")

				runningPodFound := false
				for _, pod := range podList.Items {
					if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
						runningPodFound = true
						break
					}
				}
				g.Expect(runningPodFound).Should(gomega.BeTrue(), "at least one pod should be running with IP")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Waiting for trainerStatus annotation to be populated")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist")
				g.Expect(statusJSON).ShouldNot(gomega.BeEmpty(), "trainerStatus should not be empty")

				var status progression.AnnotationStatus
				err := json.Unmarshal([]byte(statusJSON), &status)
				g.Expect(err).ShouldNot(gomega.HaveOccurred(), "trainerStatus should be valid JSON")

				// Verify essential fields (handle pointers properly)
				if status.CurrentStep != nil {
					g.Expect(*status.CurrentStep).Should(gomega.BeNumerically(">=", 0))
				}
				if status.CurrentEpoch != nil {
					g.Expect(*status.CurrentEpoch).Should(gomega.BeNumerically(">=", 0))
				}
				g.Expect(status.LastUpdatedTime).ShouldNot(gomega.BeEmpty())

				// If progress percentage is set, verify it's valid
				if status.ProgressPercentage != nil {
					g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically(">=", 0))
					g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically("<=", 100))
				}
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying trainerStatus is updated during training (or at completion)")
			gomega.Eventually(func(g gomega.Gomega) bool {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				// Check if job completed - if so, we should have at least the final status
				jobCompleted := false
				for _, cond := range gotTrainJob.Status.Conditions {
					if (cond.Type == trainer.TrainJobComplete || cond.Type == trainer.TrainJobFailed) &&
						cond.Status == metav1.ConditionTrue {
						jobCompleted = true
						break
					}
				}

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				if !exists {
					// If job completed but no annotation, fail
					g.Expect(jobCompleted).Should(gomega.BeFalse(), "Job completed but no trainerStatus annotation found")
					return false
				}

				// Verify we have valid status
				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
				g.Expect(status.LastUpdatedTime).ShouldNot(gomega.BeEmpty())

				// Success - we have a status annotation (either during or after training)
				return true
			}, timeout, interval).Should(gomega.BeTrue())

			ginkgo.By("Waiting for TrainJob to complete and verify final status")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				// Check if job is completed
				completed := false
				for _, cond := range gotTrainJob.Status.Conditions {
					if cond.Type == trainer.TrainJobComplete && cond.Status == metav1.ConditionTrue {
						completed = true
						break
					}
				}

				if completed {
					// Verify final status shows 100% progress
					statusJSON := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
					var status progression.AnnotationStatus
					g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())

					g.Expect(status.ProgressPercentage).ShouldNot(gomega.BeNil())
					g.Expect(*status.ProgressPercentage).Should(gomega.Equal(100))
					g.Expect(status.EstimatedRemainingSeconds).ShouldNot(gomega.BeNil())
					g.Expect(*status.EstimatedRemainingSeconds).Should(gomega.Equal(0))
				}

				g.Expect(completed).Should(gomega.BeTrue(), "TrainJob should complete")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When progression tracking is NOT enabled", func() {
		ginkgo.It("should NOT create trainerStatus annotation", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-disabled").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob without progression tracking annotation")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying trainerStatus annotation is NOT created (checking over time)")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
			}, 10*interval, interval).Should(gomega.Succeed())
			gomega.Consistently(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeFalse(), "trainerStatus annotation should not exist")
			}, consistentDuration, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When progression tracking annotation has invalid value", func() {
		ginkgo.It("should NOT enable progression tracking for non-'true' values", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-invalid").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "enabled").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with invalid annotation value")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying progression tracking is NOT enabled (checking over time)")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
			}, 10*interval, interval).Should(gomega.Succeed())
			gomega.Consistently(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeFalse(), "progression should not be enabled for non-true values")
			}, consistentDuration, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When metrics polling configuration is customized", func() {
		ginkgo.It("should honor custom metrics port and poll interval", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-custom-config").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "8080").
				Annotation(constants.AnnotationMetricsPollInterval, "15s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with custom metrics configuration")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying custom configuration is applied")
			gotTrainJob := &trainer.TrainJob{}
			gomega.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

			gomega.Expect(progression.GetMetricsPort(gotTrainJob)).Should(gomega.Equal("8080"))
			gomega.Expect(progression.GetMetricsPollInterval(gotTrainJob)).Should(gomega.Equal(15 * time.Second))
		})

		ginkgo.It("should handle minimum SDK-recommended poll interval (5s)", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-min-interval").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "5s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with minimum poll interval")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying minimum interval is respected")
			gotTrainJob := &trainer.TrainJob{}
			gomega.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
			gomega.Expect(progression.GetMetricsPollInterval(gotTrainJob)).Should(gomega.Equal(5 * time.Second))
		})

		ginkgo.It("should handle maximum SDK-recommended poll interval (300s)", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-max-interval").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "300s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with maximum poll interval")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying maximum interval is respected")
			gotTrainJob := &trainer.TrainJob{}
			gomega.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
			gomega.Expect(progression.GetMetricsPollInterval(gotTrainJob)).Should(gomega.Equal(300 * time.Second))
		})

		ginkgo.It("should use default interval when annotation is missing", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-default-interval").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				// No poll interval annotation - should use default (30s)
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob without poll interval annotation")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying default interval (30s) is used")
			gotTrainJob := &trainer.TrainJob{}
			gomega.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
			gomega.Expect(progression.GetMetricsPollInterval(gotTrainJob)).Should(gomega.Equal(30 * time.Second))
		})
	})

	ginkgo.Context("When TrainJob fails during training", func() {
		var failingRuntime *trainer.TrainingRuntime

		ginkgo.BeforeEach(func() {
			// Load and create failing TrainingRuntime with unique name
			runtimeFile := filepath.Join("..", "test", "resources", "failing-test-runtime.yaml")
			uniqueName := fmt.Sprintf("failing-test-runtime-%d", time.Now().UnixNano())
			var err error
			failingRuntime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(k8sClient.Create(ctx, failingRuntime)).To(gomega.Succeed())

			gomega.Eventually(func(g gomega.Gomega) {
				gotRuntime := &trainer.TrainingRuntime{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(failingRuntime), gotRuntime)).Should(gomega.Succeed())
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.DeferCleanup(func() {
				if failingRuntime != nil {
					time.Sleep(2 * time.Second)
					gomega.Expect(k8sClient.Delete(ctx, failingRuntime)).To(gomega.Succeed())
				}
			})
		})

		ginkgo.It("should capture final status even when job fails", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-job-failure").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), failingRuntime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "2s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob that will fail mid-training")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for trainerStatus annotation to be populated during training")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
				_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist during training")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob to fail")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
				failed := false
				for _, condition := range gotTrainJob.Status.Conditions {
					if condition.Type == trainer.TrainJobFailed && condition.Status == metav1.ConditionTrue {
						failed = true
						break
					}
				}
				g.Expect(failed).Should(gomega.BeTrue(), "TrainJob should fail")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying final status is captured when job fails")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist after failure")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())

				// Progress should reflect where it stopped (NOT forced to 100%)
				g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil(), "Progress should be captured")
				g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically(">=", 0), "Progress should be valid")
				g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically("<=", 100), "Progress should be valid")

				// Summary should indicate failure
				g.Expect(status.EstimatedRemainingTimeSummary).Should(gomega.ContainSubstring("failed"),
					"Summary should indicate job failed")

				g.Expect(status.LastUpdatedTime).NotTo(gomega.BeEmpty(), "LastUpdatedTime should be set")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When metrics endpoint is unreachable", func() {
		var noMetricsRuntime *trainer.TrainingRuntime

		ginkgo.BeforeEach(func() {
			// Load and create runtime without metrics endpoint with unique name
			runtimeFile := filepath.Join("..", "test", "resources", "no-metrics-runtime.yaml")
			uniqueName := fmt.Sprintf("no-metrics-runtime-%d", time.Now().UnixNano())
			var err error
			noMetricsRuntime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(k8sClient.Create(ctx, noMetricsRuntime)).To(gomega.Succeed())

			gomega.Eventually(func(g gomega.Gomega) {
				gotRuntime := &trainer.TrainingRuntime{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(noMetricsRuntime), gotRuntime)).Should(gomega.Succeed())
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.DeferCleanup(func() {
				if noMetricsRuntime != nil {
					time.Sleep(2 * time.Second)
					gomega.Expect(k8sClient.Delete(ctx, noMetricsRuntime)).To(gomega.Succeed())
				}
			})
		})

		ginkgo.It("should handle connection errors gracefully without crashing", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-no-metrics").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), noMetricsRuntime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "2s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob without metrics endpoint")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob pod to be running")
			var pod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				var err error
				pod, err = progression.GetPrimaryPod(ctx, k8sClient, trainJob)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(pod).NotTo(gomega.BeNil())
				g.Expect(pod.Status.Phase).To(gomega.Equal(corev1.PodRunning))
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying controller continues to reconcile despite connection errors")
			// Controller should log errors but continue running
			// TrainJob should not have trainerStatus annotation since metrics are unreachable during running phase
			gomega.Consistently(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				// Only check while job is still running (not completed/failed)
				isRunning := true
				for _, cond := range gotTrainJob.Status.Conditions {
					if (cond.Type == trainer.TrainJobComplete || cond.Type == trainer.TrainJobFailed) &&
						cond.Status == metav1.ConditionTrue {
						isRunning = false
						break
					}
				}

				// Annotation should not be created while running if metrics are unreachable
				// (It will be synthesized after completion, which is checked later)
				if isRunning {
					_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
					g.Expect(exists).Should(gomega.BeFalse(), "trainerStatus should not be created during running when metrics are unreachable")
				}
			}, consistentDuration, interval).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob to complete despite metrics errors")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())
				completed := false
				for _, condition := range gotTrainJob.Status.Conditions {
					if condition.Type == trainer.TrainJobComplete && condition.Status == metav1.ConditionTrue {
						completed = true
						break
					}
				}
				g.Expect(completed).Should(gomega.BeTrue(), "TrainJob should complete even without metrics")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying final status is synthesized for completed job")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "Final status should be synthesized on completion")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
				g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil())
				g.Expect(*status.ProgressPercentage).Should(gomega.Equal(100), "Final progress should be 100%")
				g.Expect(status.EstimatedRemainingTimeSummary).Should(gomega.Equal("complete"))
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("Termination Message Capture", func() {
		ginkgo.It("should capture final metrics from pod termination message", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-termination-msg").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "10s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with progression tracking enabled")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob to complete")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				completed := false
				for _, cond := range gotTrainJob.Status.Conditions {
					if cond.Type == trainer.TrainJobComplete && cond.Status == metav1.ConditionTrue {
						completed = true
						break
					}
				}
				g.Expect(completed).Should(gomega.BeTrue(), "TrainJob should complete")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying pod has termination message written by SDK")
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()), "at least one pod should exist")

				// Find the trainer container and check termination message
				foundTerminationMessage := false
				for _, pod := range podList.Items {
					for _, containerStatus := range pod.Status.ContainerStatuses {
						if containerStatus.Name != "node" && containerStatus.Name != "trainer" {
							continue
						}
						if containerStatus.State.Terminated != nil {
							message := containerStatus.State.Terminated.Message
							if message != "" {
								foundTerminationMessage = true
								// Validate it's valid JSON with expected fields
								var terminationData map[string]interface{}
								err := json.Unmarshal([]byte(message), &terminationData)
								g.Expect(err).NotTo(gomega.HaveOccurred(), "termination message should be valid JSON")
								g.Expect(terminationData).Should(gomega.HaveKey("progressPercentage"))
								break
							}
						}
					}
					if foundTerminationMessage {
						break
					}
				}
				g.Expect(foundTerminationMessage).Should(gomega.BeTrue(),
					"at least one trainer container should have termination message")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying controller captured final metrics from termination message")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())

				// Verify final status shows 100% progress (captured from termination message)
				g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil())
				g.Expect(*status.ProgressPercentage).Should(gomega.Equal(100),
					"Final progress should be 100% from termination message")
				g.Expect(status.EstimatedRemainingSeconds).NotTo(gomega.BeNil())
				g.Expect(*status.EstimatedRemainingSeconds).Should(gomega.Equal(0),
					"Remaining seconds should be 0 at completion")
				g.Expect(status.LastUpdatedTime).NotTo(gomega.BeEmpty())
			}, timeout, interval).Should(gomega.Succeed())
		})

		ginkgo.It("should fall back to HTTP polling when termination message is unavailable", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-no-termination-msg").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "5s").
				Trainer(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(intstr.FromInt(1)).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj()).
				Obj()

			ginkgo.By("Creating TrainJob with progression tracking")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for TrainJob to start and metrics to be polled via HTTP")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus should be populated via HTTP polling")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
				g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil(), "progress should be tracked via HTTP")
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying final status is captured even if termination message fails")
			// The controller should still capture final status through either:
			// 1. Termination message (if SDK wrote it)
			// 2. Last HTTP poll (fallback)
			// 3. Synthesized 100% (last resort)
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				completed := false
				for _, cond := range gotTrainJob.Status.Conditions {
					if (cond.Type == trainer.TrainJobComplete || cond.Type == trainer.TrainJobFailed) &&
						cond.Status == metav1.ConditionTrue {
						completed = true
						break
					}
				}

				if completed {
					statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
					g.Expect(exists).Should(gomega.BeTrue(), "Final status should always be captured")

					var status progression.AnnotationStatus
					g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
					g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil())
					g.Expect(*status.ProgressPercentage).Should(gomega.Equal(100))
				}

				g.Expect(completed).Should(gomega.BeTrue(), "TrainJob should complete")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})
})
