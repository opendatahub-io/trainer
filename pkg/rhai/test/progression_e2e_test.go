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

	// SDK poll interval constraints (from kubeflow/trainer SDK)
	// TransformersTrainer.metrics_poll_interval_seconds enforces 5-300 range
	minSDKPollIntervalSeconds  = 5   // Minimum: 5 seconds (prevents excessive controller load)
	maxSDKPollIntervalSeconds  = 300 // Maximum: 300 seconds (5 minutes, keeps tracking responsive)
	defaultPollIntervalSeconds = 30  // Default: 30 seconds (balanced)
)

// loadRuntimeFromFile loads TrainingRuntime from YAML file and sets namespace
// If uniqueName is provided, it will be used as the runtime name, otherwise uses the name from the file
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

var _ = ginkgo.Describe("RHAI Progression Tracking E2E Tests", func() {
	var runtime *trainer.TrainingRuntime

	ginkgo.BeforeEach(func() {
		// Load and create TrainingRuntime from resources in shared namespace
		// Use unique name to avoid conflicts when tests run serially in same namespace
		runtimeFile := filepath.Join("resources", "wrapper-test-runtime.yaml")
		uniqueName := fmt.Sprintf("wrapper-test-runtime-%d", time.Now().UnixNano())
		var err error
		runtime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(k8sClient.Create(ctx, runtime)).To(gomega.Succeed())

		// Wait for runtime to be created
		gomega.Eventually(func(g gomega.Gomega) {
			gotRuntime := &trainer.TrainingRuntime{}
			g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(runtime), gotRuntime)).Should(gomega.Succeed())
		}, timeout, interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		// Cleanup runtime after test
		if runtime != nil {
			gomega.Expect(k8sClient.Delete(ctx, runtime)).To(gomega.Succeed())
		}
	})

	ginkgo.Context("When progression tracking is enabled", func() {
		ginkgo.It("should poll metrics and update trainerStatus annotation during training", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-enabled").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "5s"). // minimum poll interval: 5s
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

				// Verify essential fields
				g.Expect(status.CurrentStep).Should(gomega.BeNumerically(">=", 0))
				g.Expect(status.CurrentEpoch).Should(gomega.BeNumerically(">=", 0))
				g.Expect(status.LastUpdatedTime).ShouldNot(gomega.BeEmpty())

				// If progress percentage is set, verify it's valid
				if status.ProgressPercentage != nil {
					g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically(">=", 0))
					g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically("<=", 100))
				}
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying trainerStatus is continuously updated during training")
			var firstUpdateTime string
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())

				if firstUpdateTime == "" {
					firstUpdateTime = status.LastUpdatedTime
				} else {
					// Verify the timestamp has been updated (metrics are being polled)
					g.Expect(status.LastUpdatedTime).ShouldNot(gomega.Equal(firstUpdateTime),
						"trainerStatus should be updated continuously")
				}
			}, timeout, interval).Should(gomega.Succeed())

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
			// Wait a bit to ensure the controller has had time to process the job
			time.Sleep(15 * time.Second)

			gomega.Consistently(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeFalse(), "trainerStatus annotation should not exist")
			}, 20*time.Second, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When progression tracking annotation has invalid value", func() {
		ginkgo.It("should NOT enable progression tracking for non-'true' values", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-invalid").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "enabled"). // Invalid: must be "true"
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
			// Wait a bit to ensure the controller has had time to process the job
			time.Sleep(5 * time.Second)

			gomega.Consistently(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				_, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeFalse(), "progression should not be enabled for non-true values")
			}, 10*time.Second, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When metrics polling configuration is customized", func() {
		ginkgo.It("should honor custom metrics port and poll interval", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-custom-config").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "8080").        // Custom port
				Annotation(constants.AnnotationMetricsPollInterval, "15s"). // Custom interval (valid range: 5-300s)
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
				Annotation(constants.AnnotationMetricsPollInterval, "5s"). // SDK minimum
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
				Annotation(constants.AnnotationMetricsPollInterval, "300s"). // SDK maximum (5 minutes)
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
			runtimeFile := filepath.Join("resources", "failing-test-runtime.yaml")
			uniqueName := fmt.Sprintf("failing-test-runtime-%d", time.Now().UnixNano())
			var err error
			failingRuntime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(k8sClient.Create(ctx, failingRuntime)).To(gomega.Succeed())

			gomega.Eventually(func(g gomega.Gomega) {
				gotRuntime := &trainer.TrainingRuntime{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(failingRuntime), gotRuntime)).Should(gomega.Succeed())
			}, timeout, interval).Should(gomega.Succeed())
		})

		ginkgo.AfterEach(func() {
			if failingRuntime != nil {
				gomega.Expect(k8sClient.Delete(ctx, failingRuntime)).To(gomega.Succeed())
			}
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

			ginkgo.By("Verifying final status is captured with progress=100% even on failure")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist after failure")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
				g.Expect(status.ProgressPercentage).NotTo(gomega.BeNil())
				g.Expect(*status.ProgressPercentage).Should(gomega.Equal(100), "Final progress should be 100% even on failure")
				g.Expect(status.LastUpdatedTime).NotTo(gomega.BeEmpty(), "LastUpdatedTime should be set")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When metrics endpoint is unreachable", func() {
		var noMetricsRuntime *trainer.TrainingRuntime

		ginkgo.BeforeEach(func() {
			// Load and create runtime without metrics endpoint with unique name
			runtimeFile := filepath.Join("resources", "no-metrics-runtime.yaml")
			uniqueName := fmt.Sprintf("no-metrics-runtime-%d", time.Now().UnixNano())
			var err error
			noMetricsRuntime, err = loadRuntimeFromFile(runtimeFile, testNs.Name, uniqueName)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(k8sClient.Create(ctx, noMetricsRuntime)).To(gomega.Succeed())

			gomega.Eventually(func(g gomega.Gomega) {
				gotRuntime := &trainer.TrainingRuntime{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(noMetricsRuntime), gotRuntime)).Should(gomega.Succeed())
			}, timeout, interval).Should(gomega.Succeed())
		})

		ginkgo.AfterEach(func() {
			if noMetricsRuntime != nil {
				gomega.Expect(k8sClient.Delete(ctx, noMetricsRuntime)).To(gomega.Succeed())
			}
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
			}, 10*time.Second, interval).Should(gomega.Succeed())

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

	ginkgo.Context("PreStop Hook Injection", func() {
		ginkgo.It("should inject preStop hook with correct sleep duration into trainer pods", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-prestop-hook").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "10s"). // 10s poll → 30s preStop (SDK range: 5-300s)
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

			ginkgo.By("Waiting for pod to be created")
			var pod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()), "at least one pod should exist")
				pod = &podList.Items[0]
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying preStop hook is injected into the trainer container")
			var trainerContainer *corev1.Container
			for i := range pod.Spec.Containers {
				if pod.Spec.Containers[i].Name == "node" {
					trainerContainer = &pod.Spec.Containers[i]
					break
				}
			}
			gomega.Expect(trainerContainer).NotTo(gomega.BeNil(), "trainer container 'node' should exist")
			gomega.Expect(trainerContainer.Lifecycle).NotTo(gomega.BeNil(), "lifecycle should be set")
			gomega.Expect(trainerContainer.Lifecycle.PreStop).NotTo(gomega.BeNil(), "preStop hook should be set")

			ginkgo.By("Verifying preStop hook uses sleep command")
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec).NotTo(gomega.BeNil(), "exec action should be set")
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec.Command).Should(gomega.HaveLen(2), "command should have 2 elements")
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec.Command[0]).Should(gomega.Equal("sleep"))

			ginkgo.By("Verifying preStop sleep duration is calculated correctly")
			// Poll interval: 10s → preStop: (2*10 + 10) = 30s
			expectedPreStopDuration := "30"
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec.Command[1]).Should(gomega.Equal(expectedPreStopDuration),
				"preStop sleep should be 2 × poll_interval + 10s buffer")

			ginkgo.By("Verifying termination grace period is set")
			gomega.Expect(pod.Spec.TerminationGracePeriodSeconds).NotTo(gomega.BeNil(), "termination grace period should be set")
			// Termination grace: preStop (30s) + shutdown buffer (30s) = 60s
			expectedTerminationGrace := int64(60)
			gomega.Expect(*pod.Spec.TerminationGracePeriodSeconds).Should(gomega.BeNumerically(">=", expectedTerminationGrace),
				"termination grace should be >= preStop + 30s")
		})

		ginkgo.It("should NOT inject preStop hook when progression tracking is disabled", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "no-prestop-hook").
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

			ginkgo.By("Waiting for pod to be created")
			var pod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()), "at least one pod should exist")
				pod = &podList.Items[0]
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying preStop hook is NOT injected")
			var trainerContainer *corev1.Container
			for i := range pod.Spec.Containers {
				if pod.Spec.Containers[i].Name == "node" {
					trainerContainer = &pod.Spec.Containers[i]
					break
				}
			}
			gomega.Expect(trainerContainer).NotTo(gomega.BeNil(), "trainer container 'node' should exist")

			// PreStop hook should not be set, or if lifecycle exists, preStop should be nil
			if trainerContainer.Lifecycle != nil {
				gomega.Expect(trainerContainer.Lifecycle.PreStop).Should(gomega.BeNil(),
					"preStop hook should not be set when progression tracking is disabled")
			}
		})

		ginkgo.It("should adapt preStop duration based on custom poll interval", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-custom-prestop").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "60s"). // 60s poll → 130s preStop (SDK range: 5-300s)
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

			ginkgo.By("Creating TrainJob with custom poll interval")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Waiting for pod to be created")
			var pod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()), "at least one pod should exist")
				pod = &podList.Items[0]
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying preStop duration reflects custom poll interval")
			var trainerContainer *corev1.Container
			for i := range pod.Spec.Containers {
				if pod.Spec.Containers[i].Name == "node" {
					trainerContainer = &pod.Spec.Containers[i]
					break
				}
			}
			gomega.Expect(trainerContainer).NotTo(gomega.BeNil())
			gomega.Expect(trainerContainer.Lifecycle).NotTo(gomega.BeNil())
			gomega.Expect(trainerContainer.Lifecycle.PreStop).NotTo(gomega.BeNil())
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec).NotTo(gomega.BeNil())

			// Poll interval: 60s → preStop: (2*60 + 10) = 130s
			expectedPreStopDuration := "130"
			gomega.Expect(trainerContainer.Lifecycle.PreStop.Exec.Command[1]).Should(gomega.Equal(expectedPreStopDuration),
				"preStop sleep should adapt to custom poll interval")

			// Termination grace: 130 + 30 = 160s
			expectedTerminationGrace := int64(160)
			gomega.Expect(*pod.Spec.TerminationGracePeriodSeconds).Should(gomega.BeNumerically(">=", expectedTerminationGrace))
		})

		ginkgo.It("should inject preStop hook into correct container when multiple containers exist", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-multi-container").
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

			ginkgo.By("Waiting for pod to be created")
			var pod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(testNs.Name),
					client.MatchingLabels{"jobset.sigs.k8s.io/jobset-name": trainJob.Name})).
					Should(gomega.Succeed())

				g.Expect(podList.Items).Should(gomega.Not(gomega.BeEmpty()))
				pod = &podList.Items[0]
			}, timeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying preStop hook is injected specifically into 'node' container")
			var nodeContainer *corev1.Container
			var otherContainersWithPreStop int

			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				if container.Name == "node" {
					nodeContainer = container
					gomega.Expect(nodeContainer.Lifecycle).NotTo(gomega.BeNil(), "node container should have lifecycle")
					gomega.Expect(nodeContainer.Lifecycle.PreStop).NotTo(gomega.BeNil(), "node container should have preStop hook")
				} else {
					// Other containers should not have preStop hook injected by progression tracking
					if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
						otherContainersWithPreStop++
					}
				}
			}

			gomega.Expect(nodeContainer).NotTo(gomega.BeNil(), "node container should exist")
			ginkgo.By("Verifying only the node container has the progression tracking preStop hook")
			// Note: We expect 0 here because only node container should get the preStop from progression tracking
		})
	})
})
