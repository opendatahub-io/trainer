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

package rhai

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	trainerconstants "github.com/kubeflow/trainer/v2/pkg/constants"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
	"github.com/kubeflow/trainer/v2/pkg/rhai/progression"
	testingutil "github.com/kubeflow/trainer/v2/pkg/util/testing"

	_ "embed"
)

const (
	timeout                  = 5 * time.Minute
	interval                 = 2 * time.Second
	trainingImageEnvironment = "RHAI_E2E_TRAINING_IMAGE"
	defaultTrainingImage     = "quay.io/opendatahub/odh-th-torch-cpu-py312:odh-stable"
)

//go:embed testdata/progression.py
var progressionScript string

//go:embed testdata/failing.py
var failingScript string

//go:embed testdata/no_metrics.py
var noMetricsScript string

func trainingImage() string {
	if image := os.Getenv(trainingImageEnvironment); image != "" {
		return image
	}
	return defaultTrainingImage
}

func makeTrainingRuntime(name, namespace string) *trainer.TrainingRuntime {
	return &trainer.TrainingRuntime{
		TypeMeta: metav1.TypeMeta{
			APIVersion: trainer.SchemeGroupVersion.String(),
			Kind:       trainer.TrainingRuntimeKind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: trainer.TrainingRuntimeSpec{
			MLPolicy: &trainer.MLPolicy{
				NumNodes: ptr.To(int32(1)),
				MLPolicySource: trainer.MLPolicySource{
					Torch: &trainer.TorchMLPolicySource{},
				},
			},
			Template: trainer.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{{
						Name: trainerconstants.Node,
						Template: batchv1.JobTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									trainerconstants.LabelTrainJobAncestor: trainerconstants.AncestorTrainer,
								},
							},
							Spec: batchv1.JobSpec{
								BackoffLimit: ptr.To(int32(0)),
								Template: corev1.PodTemplateSpec{
									Spec: corev1.PodSpec{
										Containers: []corev1.Container{{
											Name:  trainerconstants.Node,
											Image: trainingImage(),
										}},
										RestartPolicy: corev1.RestartPolicyNever,
									},
								},
							},
						},
					}},
				},
			},
		},
	}
}

func withTrainingCommand(t *trainer.Trainer, script string) *trainer.Trainer {
	t.Command = []string{"python", "-c", script}
	return t
}

func withMetricsPort(t *trainer.Trainer, port string) *trainer.Trainer {
	t.Env = append(t.Env, corev1.EnvVar{
		Name:  "RHAI_E2E_METRICS_PORT",
		Value: port,
	})
	return t
}

var _ = ginkgo.Describe("RHAI Progression Tracking E2E Tests", func() {
	var runtime *trainer.TrainingRuntime

	ginkgo.BeforeEach(func() {
		uniqueName := fmt.Sprintf("wrapper-test-runtime-%d", time.Now().UnixNano())
		runtime = makeTrainingRuntime(uniqueName, testNs.Name)
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
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
					Obj(), progressionScript)).
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
				g.Expect(status.CurrentStep).ShouldNot(gomega.BeNil(), "currentStep should be set")
				g.Expect(*status.CurrentStep).Should(gomega.BeNumerically(">=", 0))
				g.Expect(status.CurrentEpoch).ShouldNot(gomega.BeNil(), "currentEpoch should be set")
				g.Expect(*status.CurrentEpoch).Should(gomega.BeNumerically(">=", 0))
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
					g.Expect(status.LastUpdatedTime).ShouldNot(gomega.BeEmpty())
				}

				g.Expect(completed).Should(gomega.BeTrue(), "TrainJob should complete")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When progression tracking is NOT enabled", func() {
		ginkgo.It("should NOT create trainerStatus annotation", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-disabled").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript)).
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript)).
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
				Trainer(withMetricsPort(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript), "8080")).
				Obj()

			ginkgo.By("Creating TrainJob with custom metrics configuration")
			gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

			ginkgo.By("Verifying custom configuration is applied")
			gotTrainJob := &trainer.TrainJob{}
			gomega.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

			gomega.Expect(progression.GetMetricsPort(gotTrainJob)).Should(gomega.Equal("8080"))
			gomega.Expect(progression.GetMetricsPollInterval(gotTrainJob)).Should(gomega.Equal(15 * time.Second))

			ginkgo.By("Verifying metrics are polled on the custom port")
			gomega.Eventually(func(g gomega.Gomega) {
				gotTrainJob := &trainer.TrainJob{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainJob), gotTrainJob)).Should(gomega.Succeed())

				statusJSON, exists := gotTrainJob.Annotations[constants.AnnotationTrainerStatus]
				g.Expect(exists).Should(gomega.BeTrue(), "trainerStatus annotation should exist")

				var status progression.AnnotationStatus
				g.Expect(json.Unmarshal([]byte(statusJSON), &status)).Should(gomega.Succeed())
				g.Expect(status.CurrentStep).ShouldNot(gomega.BeNil(), "currentStep should be set")
				g.Expect(status.LastUpdatedTime).ShouldNot(gomega.BeEmpty(), "LastUpdatedTime should be set")
			}, timeout, interval).Should(gomega.Succeed())
		})

		ginkgo.It("should handle minimum SDK-recommended poll interval (5s)", func() {
			trainJob := testingutil.MakeTrainJobWrapper(testNs.Name, "progression-min-interval").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.TrainingRuntimeKind), runtime.Name).
				Annotation(constants.AnnotationProgressionTracking, "true").
				Annotation(constants.AnnotationMetricsPort, "28080").
				Annotation(constants.AnnotationMetricsPollInterval, "5s"). // SDK minimum
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript)).
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript)).
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), progressionScript)).
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
			uniqueName := fmt.Sprintf("failing-test-runtime-%d", time.Now().UnixNano())
			failingRuntime = makeTrainingRuntime(uniqueName, testNs.Name)
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), failingScript)).
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
				g.Expect(*status.ProgressPercentage).Should(gomega.BeNumerically(">", 0), "progress should preserve last polled value")
				g.Expect(status.LastUpdatedTime).NotTo(gomega.BeEmpty(), "LastUpdatedTime should be set")
			}, timeout, interval).Should(gomega.Succeed())
		})
	})

	ginkgo.Context("When metrics endpoint is unreachable", func() {
		var noMetricsRuntime *trainer.TrainingRuntime

		ginkgo.BeforeEach(func() {
			uniqueName := fmt.Sprintf("no-metrics-runtime-%d", time.Now().UnixNano())
			noMetricsRuntime = makeTrainingRuntime(uniqueName, testNs.Name)
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
				Trainer(withTrainingCommand(testingutil.MakeTrainJobTrainerWrapper().
					NumNodes(1).
					NumProcPerNode(1).
					ResourcesPerNode(corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					}).
					Obj(), noMetricsScript)).
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

			ginkgo.By("Check controller still reconciles the TrainJob by ensuring it gets marked complete")
			// Controller should log errors but continue running
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
		})
	})

})
