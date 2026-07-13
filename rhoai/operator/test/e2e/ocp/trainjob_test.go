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

package ocp

import (
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

const trainerNamespace = "opendatahub"

func TestTrainJobWithClusterTrainingRuntime(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	const (
		testTrainJobName = "test-trainjob-workflow"
		targetCTR        = "torch-distributed-cpu"
	)

	err := k8sClient.CreateTrainer(ctx, trainerNamespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	t.Cleanup(func() {
		_ = k8sClient.DeleteTrainJob(
			ctx, testTrainJobName, trainerNamespace)
		_ = k8sClient.DeleteTrainer(ctx)
	})

	g.Eventually(func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.Phase).To(Equal(common.PhaseReady))
	}).Should(Succeed())

	_, err = k8sClient.GetClusterTrainingRuntime(ctx, targetCTR)
	g.Expect(err).NotTo(HaveOccurred(),
		"ClusterTrainingRuntime %s should exist", targetCTR)

	err = k8sClient.CreateTrainJobWithCommand(
		ctx, testTrainJobName, trainerNamespace, targetCTR,
		[]string{
			"python", "-c",
			"import torch; " +
				"print(f'PyTorch version: {torch.__version__}'); " +
				"print('Training completed successfully')",
		},
	)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")

	g.Eventually(func(g Gomega) {
		jobSets, err := k8sClient.ListJobSets(ctx, trainerNamespace)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(jobSets).To(ContainElement(testTrainJobName),
			"Trainer controller should create a JobSet")
	}).Should(Succeed())

	g.Eventually(func(g Gomega) {
		tj, err := k8sClient.GetTrainJob(
			ctx, testTrainJobName, trainerNamespace)
		g.Expect(err).NotTo(HaveOccurred())
		if trainJobFailed(tj) {
			StopTrying("TrainJob reached terminal Failed state").
				Wrap(fmt.Errorf("conditions: %v",
					trainJobConditions(tj))).Now()
		}
		g.Expect(trainJobCompleted(tj)).To(BeTrue(),
			"TrainJob should complete successfully, "+
				"conditions: %v", trainJobConditions(tj))
	}).WithTimeout(20 * time.Minute).Should(Succeed())
}

func trainJobCompleted(tj *unstructured.Unstructured) bool {
	return trainJobHasCondition(tj, "Complete")
}

func trainJobFailed(tj *unstructured.Unstructured) bool {
	return trainJobHasCondition(tj, "Failed")
}

func trainJobHasCondition(
	tj *unstructured.Unstructured, condType string,
) bool {
	for _, c := range trainJobConditions(tj) {
		cond, _ := c.(map[string]any)
		if cond["type"] == condType &&
			cond["status"] == string(metav1.ConditionTrue) {
			return true
		}
	}
	return false
}

func trainJobConditions(
	tj *unstructured.Unstructured,
) []any {
	conditions, _, _ := unstructured.NestedSlice(
		tj.Object, "status", "conditions")
	return conditions
}
