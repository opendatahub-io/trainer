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

package webhooks

import (
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	testingutil "github.com/kubeflow/trainer/v2/pkg/util/testing"
	"github.com/kubeflow/trainer/v2/test/integration/framework"
	"github.com/kubeflow/trainer/v2/test/util"
)

var _ = ginkgo.Describe("TrainJob RHOAI Security Webhook", ginkgo.Ordered, func() {
	var ns *corev1.Namespace
	var clusterTrainingRuntime *trainer.ClusterTrainingRuntime
	runtimeName := "rhoai-security-runtime"

	ginkgo.BeforeAll(func() {
		fwk = &framework.Framework{}
		cfg = fwk.Init()
		ctx, k8sClient = fwk.RunManager(cfg, false)
	})
	ginkgo.AfterAll(func() {
		fwk.Teardown()
	})

	ginkgo.BeforeEach(func() {
		ns = &corev1.Namespace{
			TypeMeta: metav1.TypeMeta{
				APIVersion: corev1.SchemeGroupVersion.String(),
				Kind:       "Namespace",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "trainjob-rhoai-",
			},
		}
		gomega.Expect(k8sClient.Create(ctx, ns)).To(gomega.Succeed())

		clusterTrainingRuntime = testingutil.MakeClusterTrainingRuntimeWrapper(runtimeName).RuntimeSpec(
			testingutil.MakeTrainingRuntimeSpecWrapper(
				testingutil.MakeClusterTrainingRuntimeWrapper(runtimeName).Spec).Obj()).Obj()
		gomega.Expect(k8sClient.Create(ctx, clusterTrainingRuntime)).To(gomega.Succeed())
		gomega.Eventually(func(g gomega.Gomega) {
			g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(clusterTrainingRuntime), clusterTrainingRuntime)).Should(gomega.Succeed())
		}, util.Timeout, util.Interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		gomega.Expect(k8sClient.DeleteAllOf(ctx, &trainer.TrainJob{}, client.InNamespace(ns.Name))).To(gomega.Succeed())
		gomega.Expect(k8sClient.DeleteAllOf(ctx, &trainer.ClusterTrainingRuntime{})).To(gomega.Succeed())
	})

	runtimePatchWithTolerations := func(tolerations ...corev1.Toleration) []trainer.RuntimePatch {
		return []trainer.RuntimePatch{
			{
				Manager: "acme.io/security-test",
				TrainingRuntimeSpec: &trainer.TrainingRuntimeSpecPatch{
					Template: &trainer.JobSetTemplatePatch{
						Spec: &trainer.JobSetSpecPatch{
							ReplicatedJobs: []trainer.ReplicatedJobPatch{{
								Name: "node",
								Template: &trainer.JobTemplatePatch{
									Spec: &trainer.JobSpecPatch{
										Template: &trainer.PodTemplatePatch{
											Spec: &trainer.PodSpecPatch{
												Tolerations: tolerations,
											},
										},
									},
								},
							}},
						},
					},
				},
			},
		}
	}

	ginkgo.When("Creating TrainJob with RHOAI security validations", func() {
		ginkgo.DescribeTable("Validate toleration security on creation", func(trainJob func() *trainer.TrainJob, errorMatcher gomega.OmegaMatcher) {
			gomega.Expect(k8sClient.Create(ctx, trainJob())).Should(errorMatcher)
		},
			ginkgo.Entry("Should succeed with safe GPU toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "safe-toleration").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						})).
						Obj()
				},
				gomega.Succeed()),
			ginkgo.Entry("Should succeed without any runtime patches",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "no-patches").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						Obj()
				},
				gomega.Succeed()),
			ginkgo.Entry("Should fail with control-plane toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "control-plane-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "node-role.kubernetes.io/control-plane",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						})).
						Obj()
				},
				testingutil.BeForbiddenError()),
			ginkgo.Entry("Should fail with master node toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "master-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "node-role.kubernetes.io/master",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						})).
						Obj()
				},
				testingutil.BeForbiddenError()),
			ginkgo.Entry("Should fail with infra node toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "infra-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "node-role.kubernetes.io/infra",
							Operator: corev1.TolerationOpEqual,
							Value:    "",
						})).
						Obj()
				},
				testingutil.BeForbiddenError()),
			ginkgo.Entry("Should fail with wildcard Exists toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "wildcard-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Operator: corev1.TolerationOpExists,
						})).
						Obj()
				},
				testingutil.BeForbiddenError()),
		)
	})

	ginkgo.When("Updating TrainJob with RHOAI security validations", func() {
		ginkgo.DescribeTable("Validate toleration security on update", func(old func() *trainer.TrainJob, update func(*trainer.TrainJob) *trainer.TrainJob, errorMatcher gomega.OmegaMatcher) {
			oldTrainJob := old()
			gomega.Expect(k8sClient.Create(ctx, oldTrainJob)).Should(gomega.Succeed())
			gomega.Eventually(func(g gomega.Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(oldTrainJob), oldTrainJob)).Should(gomega.Succeed())
				g.Expect(k8sClient.Update(ctx, update(oldTrainJob))).Should(errorMatcher)
			}, util.Timeout, util.Interval).Should(gomega.Succeed())
		},
			ginkgo.Entry("Should fail to update TrainJob adding control-plane toleration",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "update-cp-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						Suspend(true).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
						})).
						Obj()
				},
				func(job *trainer.TrainJob) *trainer.TrainJob {
					job.Spec.RuntimePatches[0].TrainingRuntimeSpec.Template.Spec.ReplicatedJobs[0].
						Template.Spec.Template.Spec.Tolerations = []corev1.Toleration{
						{
							Key:      "node-role.kubernetes.io/control-plane",
							Operator: corev1.TolerationOpExists,
						},
					}
					return job
				},
				testingutil.BeForbiddenError()),
			ginkgo.Entry("Should succeed to update TrainJob keeping safe tolerations",
				func() *trainer.TrainJob {
					return testingutil.MakeTrainJobWrapper(ns.Name, "update-safe-tol").
						RuntimeRef(trainer.GroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), runtimeName).
						Suspend(true).
						RuntimePatches(runtimePatchWithTolerations(corev1.Toleration{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
						})).
						Obj()
				},
				func(job *trainer.TrainJob) *trainer.TrainJob {
					job.Spec.RuntimePatches[0].TrainingRuntimeSpec.Template.Spec.ReplicatedJobs[0].
						Template.Spec.Template.Spec.NodeSelector = map[string]string{"zone": "us-east"}
					return job
				},
				gomega.Succeed()),
		)
	})
})
