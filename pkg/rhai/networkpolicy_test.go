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
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
)

func TestGetNetworkPolicyName(t *testing.T) {
	tests := []struct {
		name         string
		trainJobName string
		want         string
	}{
		{
			name:         "simple name",
			trainJobName: "my-training-job",
			want:         "my-training-job",
		},
		{
			name:         "short name",
			trainJobName: "job",
			want:         "job",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trainJob := &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name: tt.trainJobName,
				},
			}
			got := getNetworkPolicyName(trainJob)
			if got != tt.want {
				t.Errorf("getNetworkPolicyName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuildNetworkPolicy(t *testing.T) {
	tests := []struct {
		name                   string
		trainJob               *trainer.TrainJob
		wantName               string
		wantNamespace          string
		wantMetricsPort        int
		wantJobSelector        string
		wantOwnerRefName       string
		wantIngressRules       int
		wantMetricsRulePresent bool
	}{
		{
			name: "progression enabled - default metrics port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "user-namespace",
					UID:       types.UID("test-uid-123"),
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "true",
					},
				},
			},
			wantName:               "test-job",
			wantNamespace:          "user-namespace",
			wantMetricsPort:        28080,
			wantJobSelector:        "test-job",
			wantOwnerRefName:       "test-job",
			wantIngressRules:       2,
			wantMetricsRulePresent: true,
		},
		{
			name: "progression enabled - custom metrics port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "custom-port-job",
					Namespace: "ml-workloads",
					UID:       types.UID("uid-456"),
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "true",
						constants.AnnotationMetricsPort:         "8080",
					},
				},
			},
			wantName:               "custom-port-job",
			wantNamespace:          "ml-workloads",
			wantMetricsPort:        8080,
			wantJobSelector:        "custom-port-job",
			wantOwnerRefName:       "custom-port-job",
			wantIngressRules:       2,
			wantMetricsRulePresent: true,
		},
		{
			name: "progression disabled - only pod isolation rule",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "no-progression-job",
					Namespace: "default",
					UID:       types.UID("uid-no-prog"),
				},
			},
			wantName:               "no-progression-job",
			wantNamespace:          "default",
			wantJobSelector:        "no-progression-job",
			wantOwnerRefName:       "no-progression-job",
			wantIngressRules:       1,
			wantMetricsRulePresent: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := buildNetworkPolicy(tt.trainJob)

			// Verify metadata
			if policy.Name != tt.wantName {
				t.Errorf("Name = %q, want %q", policy.Name, tt.wantName)
			}
			if policy.Namespace != tt.wantNamespace {
				t.Errorf("Namespace = %q, want %q", policy.Namespace, tt.wantNamespace)
			}

			// Verify labels
			if policy.Labels["trainer.kubeflow.org/trainjob-name"] != tt.trainJob.Name {
				t.Errorf("Label trainjob-name = %q, want %q",
					policy.Labels["trainer.kubeflow.org/trainjob-name"], tt.trainJob.Name)
			}
			if policy.Labels["trainer.kubeflow.org/component"] != "network-policy" {
				t.Errorf("Label component = %q, want %q",
					policy.Labels["trainer.kubeflow.org/component"], "network-policy")
			}

			// Verify OwnerReference
			if len(policy.OwnerReferences) != 1 {
				t.Fatalf("Expected 1 OwnerReference, got %d", len(policy.OwnerReferences))
			}
			ownerRef := policy.OwnerReferences[0]
			if ownerRef.Name != tt.wantOwnerRefName {
				t.Errorf("OwnerReference.Name = %q, want %q", ownerRef.Name, tt.wantOwnerRefName)
			}
			if ownerRef.Kind != "TrainJob" {
				t.Errorf("OwnerReference.Kind = %q, want TrainJob", ownerRef.Kind)
			}
			if ownerRef.Controller == nil || !*ownerRef.Controller {
				t.Error("OwnerReference.Controller should be true")
			}
			if ownerRef.BlockOwnerDeletion == nil || !*ownerRef.BlockOwnerDeletion {
				t.Error("OwnerReference.BlockOwnerDeletion should be true")
			}

			// Verify PodSelector selects TrainJob pods
			podSelector := policy.Spec.PodSelector.MatchLabels
			if podSelector["jobset.sigs.k8s.io/jobset-name"] != tt.wantJobSelector {
				t.Errorf("PodSelector jobset-name = %q, want %q",
					podSelector["jobset.sigs.k8s.io/jobset-name"], tt.wantJobSelector)
			}

			// Verify PolicyTypes
			if len(policy.Spec.PolicyTypes) != 1 || policy.Spec.PolicyTypes[0] != networkingv1.PolicyTypeIngress {
				t.Errorf("PolicyTypes = %v, want [Ingress]", policy.Spec.PolicyTypes)
			}

			// Verify Ingress rules count
			if len(policy.Spec.Ingress) != tt.wantIngressRules {
				t.Fatalf("Expected %d ingress rules, got %d", tt.wantIngressRules, len(policy.Spec.Ingress))
			}

			// Find the pod isolation rule (always present, no ports restriction)
			var podIsolationRule *networkingv1.NetworkPolicyIngressRule
			var metricsRule *networkingv1.NetworkPolicyIngressRule
			for i := range policy.Spec.Ingress {
				rule := &policy.Spec.Ingress[i]
				if len(rule.Ports) == 0 {
					podIsolationRule = rule
				} else {
					metricsRule = rule
				}
			}

			// Verify metrics rule (only when progression enabled)
			if tt.wantMetricsRulePresent {
				if metricsRule == nil {
					t.Fatal("Expected metrics rule but not found")
				}
				if len(metricsRule.From) != 1 {
					t.Fatalf("Metrics rule: Expected 1 peer, got %d", len(metricsRule.From))
				}
				if len(metricsRule.Ports) != 1 {
					t.Fatalf("Metrics rule: Expected 1 port, got %d", len(metricsRule.Ports))
				}
				expectedPort := intstr.FromInt(tt.wantMetricsPort)
				if metricsRule.Ports[0].Port == nil || *metricsRule.Ports[0].Port != expectedPort {
					t.Errorf("Metrics rule: Port = %v, want %v", metricsRule.Ports[0].Port, expectedPort)
				}
				controllerPeer := metricsRule.From[0]
				if controllerPeer.PodSelector.MatchLabels[constants.ControllerPodLabelName] != constants.ControllerPodLabelNameValue {
					t.Errorf("Metrics rule: Controller name label = %q, want %s",
						controllerPeer.PodSelector.MatchLabels[constants.ControllerPodLabelName], constants.ControllerPodLabelNameValue)
				}
			} else {
				if metricsRule != nil {
					t.Error("Expected no metrics rule when progression disabled")
				}
			}

			// Verify pod isolation rule (always present)
			if podIsolationRule == nil {
				t.Fatal("Pod isolation rule not found")
			}
			if len(podIsolationRule.From) != 1 {
				t.Fatalf("Pod isolation rule: Expected 1 peer, got %d", len(podIsolationRule.From))
			}
			if len(podIsolationRule.Ports) != 0 {
				t.Errorf("Pod isolation rule: Expected 0 ports (all ports), got %d", len(podIsolationRule.Ports))
			}
			sameJobPeer := podIsolationRule.From[0]
			if sameJobPeer.PodSelector == nil {
				t.Fatal("Pod isolation rule: PodSelector is nil")
			}
			if sameJobPeer.PodSelector.MatchLabels["jobset.sigs.k8s.io/jobset-name"] != tt.trainJob.Name {
				t.Errorf("Pod isolation rule: Same-job selector = %q, want %q",
					sameJobPeer.PodSelector.MatchLabels["jobset.sigs.k8s.io/jobset-name"], tt.trainJob.Name)
			}
			if sameJobPeer.NamespaceSelector != nil {
				t.Error("Pod isolation rule: Should not have NamespaceSelector")
			}
		})
	}
}

func TestReconcileNetworkPolicy(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = trainer.AddToScheme(scheme)
	_ = networkingv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	tests := []struct {
		name              string
		trainJob          *trainer.TrainJob
		existingPolicy    *networkingv1.NetworkPolicy
		wantPolicyCreated bool
		wantPolicyUpdated bool
		wantErr           bool
	}{
		{
			name: "creates new NetworkPolicy",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "new-job",
					Namespace: "default",
					UID:       types.UID("uid-new"),
				},
			},
			existingPolicy:    nil,
			wantPolicyCreated: true,
			wantPolicyUpdated: false,
			wantErr:           false,
		},
		{
			name: "updates existing NetworkPolicy",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "existing-job",
					Namespace: "default",
					UID:       types.UID("uid-existing"),
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "true",
						constants.AnnotationMetricsPort:         "9090", // Changed port
					},
				},
			},
			existingPolicy: &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "existing-job",
					Namespace: "default",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"jobset.sigs.k8s.io/jobset-name": "existing-job",
						},
					},
				},
			},
			wantPolicyCreated: false,
			wantPolicyUpdated: true,
			wantErr:           false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build client with existing objects
			clientBuilder := fake.NewClientBuilder().WithScheme(scheme)
			if tt.existingPolicy != nil {
				clientBuilder = clientBuilder.WithObjects(tt.existingPolicy)
			}
			fakeClient := clientBuilder.Build()

			ctx := context.Background()
			err := ReconcileNetworkPolicy(ctx, fakeClient, tt.trainJob)

			if (err != nil) != tt.wantErr {
				t.Errorf("ReconcileNetworkPolicy() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check if policy exists
			policyName := getNetworkPolicyName(tt.trainJob)
			policy := &networkingv1.NetworkPolicy{}
			getErr := fakeClient.Get(ctx, client.ObjectKey{
				Namespace: tt.trainJob.Namespace,
				Name:      policyName,
			}, policy)

			if tt.wantPolicyCreated {
				if getErr != nil {
					t.Errorf("Expected NetworkPolicy to be created, but Get failed: %v", getErr)
					return
				}
				// Verify policy has correct owner reference
				if len(policy.OwnerReferences) != 1 {
					t.Errorf("Expected 1 OwnerReference, got %d", len(policy.OwnerReferences))
				}
			}

			if tt.wantPolicyUpdated {
				if getErr != nil {
					t.Errorf("Expected NetworkPolicy to exist for update, but Get failed: %v", getErr)
					return
				}
				// Find metrics rule (has ports) and verify port was updated
				var metricsRule *networkingv1.NetworkPolicyIngressRule
				for i := range policy.Spec.Ingress {
					if len(policy.Spec.Ingress[i].Ports) > 0 {
						metricsRule = &policy.Spec.Ingress[i]
						break
					}
				}
				if metricsRule == nil {
					t.Error("Expected metrics rule after update, but not found")
				} else {
					expectedPort := intstr.FromInt(9090)
					if *metricsRule.Ports[0].Port != expectedPort {
						t.Errorf("Expected port 9090 after update, got %v", metricsRule.Ports[0].Port)
					}
				}
			}
		})
	}
}

func TestBuildNetworkPolicy_SecurityProperties(t *testing.T) {
	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "security-test",
			Namespace: "user-namespace",
			UID:       types.UID("security-uid"),
			Annotations: map[string]string{
				constants.AnnotationProgressionTracking: "true",
				constants.AnnotationMetricsPort:         "28080",
			},
		},
	}

	policy := buildNetworkPolicy(trainJob)

	// Find rules by type
	var metricsRule, podIsolationRule *networkingv1.NetworkPolicyIngressRule
	for i := range policy.Spec.Ingress {
		rule := &policy.Spec.Ingress[i]
		if len(rule.Ports) > 0 {
			metricsRule = rule
		} else {
			podIsolationRule = rule
		}
	}

	t.Run("only allows Ingress policy type", func(t *testing.T) {
		if len(policy.Spec.PolicyTypes) != 1 {
			t.Fatalf("Expected 1 PolicyType, got %d", len(policy.Spec.PolicyTypes))
		}
		if policy.Spec.PolicyTypes[0] != networkingv1.PolicyTypeIngress {
			t.Errorf("PolicyType = %v, want Ingress", policy.Spec.PolicyTypes[0])
		}
	})

	t.Run("metrics port only accessible by controller", func(t *testing.T) {
		if metricsRule == nil {
			t.Fatal("Metrics rule not found")
		}
		if len(metricsRule.From) != 1 {
			t.Fatalf("Expected 1 peer for metrics rule, got %d", len(metricsRule.From))
		}
		if len(metricsRule.Ports) != 1 {
			t.Fatalf("Expected 1 port restriction, got %d", len(metricsRule.Ports))
		}

		peer := metricsRule.From[0]
		if peer.PodSelector.MatchLabels[constants.ControllerPodLabelName] != constants.ControllerPodLabelNameValue {
			t.Error("Missing trainer name label requirement")
		}
		if peer.PodSelector.MatchLabels[constants.ControllerPodLabelComponent] != constants.ControllerPodLabelComponentValue {
			t.Error("Missing controller component label requirement")
		}
	})

	t.Run("same-job pods cannot be spoofed from other namespaces", func(t *testing.T) {
		if podIsolationRule == nil {
			t.Fatal("Pod isolation rule not found")
		}
		peer := podIsolationRule.From[0]
		if peer.NamespaceSelector != nil {
			t.Error("Same-job rule should NOT have NamespaceSelector (must be same namespace)")
		}
	})

	t.Run("ownerReference ensures cleanup on TrainJob deletion", func(t *testing.T) {
		if len(policy.OwnerReferences) != 1 {
			t.Fatalf("Expected 1 OwnerReference, got %d", len(policy.OwnerReferences))
		}
		ref := policy.OwnerReferences[0]
		if ref.UID != trainJob.UID {
			t.Errorf("OwnerReference UID = %v, want %v", ref.UID, trainJob.UID)
		}
		if ref.Controller == nil || !*ref.Controller {
			t.Error("OwnerReference should be controller reference")
		}
	})
}
