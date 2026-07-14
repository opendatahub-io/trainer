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
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

func validateRHOAISecurity(trainJob *trainer.TrainJob) field.ErrorList {
	var allErrs field.ErrorList
	patchesPath := field.NewPath("spec", "runtimePatches")

	for i, patch := range trainJob.Spec.RuntimePatches {
		if patch.TrainingRuntimeSpec == nil || patch.TrainingRuntimeSpec.Template == nil ||
			patch.TrainingRuntimeSpec.Template.Spec == nil {
			continue
		}
		for j, rj := range patch.TrainingRuntimeSpec.Template.Spec.ReplicatedJobs {
			if rj.Template == nil || rj.Template.Spec == nil ||
				rj.Template.Spec.Template == nil || rj.Template.Spec.Template.Spec == nil {
				continue
			}
			podSpec := rj.Template.Spec.Template.Spec
			specPath := patchesPath.Index(i).Child("trainingRuntimeSpec", "template", "spec",
				"replicatedJobs").Index(j).Child("template", "spec", "template", "spec")

			for k, tol := range podSpec.Tolerations {
				if isControlPlaneToleration(tol) {
					allErrs = append(allErrs, field.Forbidden(
						specPath.Child("tolerations").Index(k),
						"tolerations targeting control-plane or master nodes are not allowed: they bypass node isolation",
					))
				}
			}
		}
	}
	return allErrs
}

func isControlPlaneToleration(tol corev1.Toleration) bool {
	dangerousKeys := []string{
		"node-role.kubernetes.io/control-plane",
		"node-role.kubernetes.io/master",
		"node-role.kubernetes.io/infra",
	}
	if slices.Contains(dangerousKeys, tol.Key) {
		return true
	}
	if tol.Operator == corev1.TolerationOpExists && tol.Key == "" {
		return true
	}
	return false
}
