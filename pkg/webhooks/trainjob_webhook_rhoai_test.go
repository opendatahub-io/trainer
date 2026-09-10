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
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	testingutil "github.com/kubeflow/trainer/v2/pkg/util/testing"
)

func TestValidateRHOAISecurity(t *testing.T) {
	cases := map[string]struct {
		obj       *trainer.TrainJob
		wantError bool
		wantCount int
	}{
		"no runtime patches": {
			obj: testingutil.MakeTrainJobWrapper("default", "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				Obj(),
		},
		"patch with nil TrainingRuntimeSpec": {
			obj: testingutil.MakeTrainJobWrapper("default", "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				RuntimePatches([]trainer.RuntimePatch{
					{Manager: "acme.io/one"},
				}).Obj(),
		},
		"patch with nil template spec": {
			obj: testingutil.MakeTrainJobWrapper("default", "test").
				RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
				RuntimePatches([]trainer.RuntimePatch{
					{
						Manager:             "acme.io/one",
						TrainingRuntimeSpec: &trainer.TrainingRuntimeSpecPatch{},
					},
				}).Obj(),
		},
		"safe toleration allowed": {
			obj: trainJobWithTolerations(corev1.Toleration{
				Key:      "nvidia.com/gpu",
				Operator: corev1.TolerationOpExists,
				Effect:   corev1.TaintEffectNoSchedule,
			}),
		},
		"control-plane toleration rejected": {
			obj: trainJobWithTolerations(corev1.Toleration{
				Key:      "node-role.kubernetes.io/control-plane",
				Operator: corev1.TolerationOpExists,
				Effect:   corev1.TaintEffectNoSchedule,
			}),
			wantError: true,
			wantCount: 1,
		},
		"master toleration rejected": {
			obj: trainJobWithTolerations(corev1.Toleration{
				Key:      "node-role.kubernetes.io/master",
				Operator: corev1.TolerationOpExists,
				Effect:   corev1.TaintEffectNoSchedule,
			}),
			wantError: true,
			wantCount: 1,
		},
		"infra toleration rejected": {
			obj: trainJobWithTolerations(corev1.Toleration{
				Key:      "node-role.kubernetes.io/infra",
				Operator: corev1.TolerationOpEqual,
				Value:    "",
			}),
			wantError: true,
			wantCount: 1,
		},
		"wildcard toleration rejected": {
			obj: trainJobWithTolerations(corev1.Toleration{
				Operator: corev1.TolerationOpExists,
			}),
			wantError: true,
			wantCount: 1,
		},
		"multiple dangerous tolerations all rejected": {
			obj: trainJobWithTolerations(
				corev1.Toleration{
					Key:      "node-role.kubernetes.io/control-plane",
					Operator: corev1.TolerationOpExists,
				},
				corev1.Toleration{
					Key:      "node-role.kubernetes.io/master",
					Operator: corev1.TolerationOpExists,
				},
			),
			wantError: true,
			wantCount: 2,
		},
		"mix of safe and dangerous tolerations": {
			obj: trainJobWithTolerations(
				corev1.Toleration{
					Key:      "nvidia.com/gpu",
					Operator: corev1.TolerationOpExists,
				},
				corev1.Toleration{
					Key:      "node-role.kubernetes.io/control-plane",
					Operator: corev1.TolerationOpExists,
				},
			),
			wantError: true,
			wantCount: 1,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			errs := validateRHOAISecurity(tc.obj)
			if tc.wantError {
				if len(errs) == 0 {
					t.Error("expected validation errors, got none")
				}
				if len(errs) != tc.wantCount {
					t.Errorf("expected %d errors, got %d: %v", tc.wantCount, len(errs), errs)
				}
				for _, err := range errs {
					if err.Type != field.ErrorTypeForbidden {
						t.Errorf("expected Forbidden error type, got %v", err.Type)
					}
				}
			} else {
				if len(errs) != 0 {
					t.Errorf("expected no errors, got %v", errs)
				}
			}
		})
	}
}

func TestIsControlPlaneToleration(t *testing.T) {
	cases := map[string]struct {
		tol  corev1.Toleration
		want bool
	}{
		"control-plane key": {
			tol:  corev1.Toleration{Key: "node-role.kubernetes.io/control-plane"},
			want: true,
		},
		"master key": {
			tol:  corev1.Toleration{Key: "node-role.kubernetes.io/master"},
			want: true,
		},
		"infra key": {
			tol:  corev1.Toleration{Key: "node-role.kubernetes.io/infra"},
			want: true,
		},
		"wildcard Exists with empty key": {
			tol:  corev1.Toleration{Operator: corev1.TolerationOpExists, Key: ""},
			want: true,
		},
		"Exists with non-empty key": {
			tol:  corev1.Toleration{Operator: corev1.TolerationOpExists, Key: "nvidia.com/gpu"},
			want: false,
		},
		"safe toleration": {
			tol:  corev1.Toleration{Key: "some-custom-key", Operator: corev1.TolerationOpEqual, Value: "true"},
			want: false,
		},
		"empty toleration with Equal operator": {
			tol:  corev1.Toleration{Operator: corev1.TolerationOpEqual, Key: ""},
			want: false,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := isControlPlaneToleration(tc.tol)
			if got != tc.want {
				t.Errorf("isControlPlaneToleration(%+v) = %v, want %v", tc.tol, got, tc.want)
			}
		})
	}
}

func trainJobWithTolerations(tolerations ...corev1.Toleration) *trainer.TrainJob {
	return testingutil.MakeTrainJobWrapper("default", "test").
		RuntimeRef(trainer.SchemeGroupVersion.WithKind(trainer.ClusterTrainingRuntimeKind), "test-runtime").
		RuntimePatches([]trainer.RuntimePatch{
			{
				Manager: "acme.io/one",
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
		}).Obj()
}
