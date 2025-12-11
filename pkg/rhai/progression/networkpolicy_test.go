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

package progression

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
			want:         "my-training-job" + constants.NetworkPolicyNameSuffix,
		},
		{
			name:         "short name",
			trainJobName: "job",
			want:         "job" + constants.NetworkPolicyNameSuffix,
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
		name             string
		trainJob         *trainer.TrainJob
		wantName         string
		wantNamespace    string
		wantMetricsPort  int
		wantJobSelector  string
		wantOwnerRefName string
		wantIngressRules int
	}{
		{
			name: "default metrics port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "user-namespace",
					UID:       types.UID("test-uid-123"),
				},
			},
			wantName:         "test-job" + constants.NetworkPolicyNameSuffix,
			wantNamespace:    "user-namespace",
			wantMetricsPort:  28080,
			wantJobSelector:  "test-job",
			wantOwnerRefName: "test-job",
			wantIngressRules: 2,
		},
		{
			name: "custom metrics port",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "custom-port-job",
					Namespace: "ml-workloads",
					UID:       types.UID("uid-456"),
					Annotations: map[string]string{
						constants.AnnotationMetricsPort: "8080",
					},
				},
			},
			wantName:         "custom-port-job" + constants.NetworkPolicyNameSuffix,
			wantNamespace:    "ml-workloads",
			wantMetricsPort:  8080,
			wantJobSelector:  "custom-port-job",
			wantOwnerRefName: "custom-port-job",
			wantIngressRules: 2,
		},
		{
			name: "invalid port falls back to default",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "invalid-port-job",
					Namespace: "default",
					UID:       types.UID("uid-789"),
					Annotations: map[string]string{
						constants.AnnotationMetricsPort: "not-a-number",
					},
				},
			},
			wantName:         "invalid-port-job" + constants.NetworkPolicyNameSuffix,
			wantNamespace:    "default",
			wantMetricsPort:  28080,
			wantJobSelector:  "invalid-port-job",
			wantOwnerRefName: "invalid-port-job",
			wantIngressRules: 2,
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
			if policy.Labels["trainer.kubeflow.org/component"] != "metrics-security" {
				t.Errorf("Label component = %q, want %q",
					policy.Labels["trainer.kubeflow.org/component"], "metrics-security")
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

			// Verify Rule 1: Controller access to metrics port
			rule1 := policy.Spec.Ingress[0]
			if len(rule1.From) != 1 {
				t.Fatalf("Rule 1: Expected 1 peer, got %d", len(rule1.From))
			}
			if len(rule1.Ports) != 1 {
				t.Fatalf("Rule 1: Expected 1 port, got %d", len(rule1.Ports))
			}

			// Check metrics port
			expectedPort := intstr.FromInt(tt.wantMetricsPort)
			if rule1.Ports[0].Port == nil || *rule1.Ports[0].Port != expectedPort {
				t.Errorf("Rule 1: Port = %v, want %v", rule1.Ports[0].Port, expectedPort)
			}
			if rule1.Ports[0].Protocol == nil || *rule1.Ports[0].Protocol != corev1.ProtocolTCP {
				t.Errorf("Rule 1: Protocol = %v, want TCP", rule1.Ports[0].Protocol)
			}

			// Check controller pod selector
			controllerPeer := rule1.From[0]
			if controllerPeer.PodSelector == nil {
				t.Fatal("Rule 1: PodSelector is nil")
			}
			if controllerPeer.PodSelector.MatchLabels["app.kubernetes.io/name"] != "trainer" {
				t.Errorf("Rule 1: Controller name label = %q, want trainer",
					controllerPeer.PodSelector.MatchLabels["app.kubernetes.io/name"])
			}
			if controllerPeer.PodSelector.MatchLabels["app.kubernetes.io/component"] != "controller" {
				t.Errorf("Rule 1: Controller component label = %q, want manager",
					controllerPeer.PodSelector.MatchLabels["app.kubernetes.io/component"])
			}

			// Check namespace selector
			if controllerPeer.NamespaceSelector == nil {
				t.Fatal("Rule 1: NamespaceSelector is nil")
			}
			if _, ok := controllerPeer.NamespaceSelector.MatchLabels["kubernetes.io/metadata.name"]; !ok {
				t.Error("Rule 1: NamespaceSelector missing kubernetes.io/metadata.name label")
			}

			// Verify Rule 2: Same-job pods access to all ports
			rule2 := policy.Spec.Ingress[1]
			if len(rule2.From) != 1 {
				t.Fatalf("Rule 2: Expected 1 peer, got %d", len(rule2.From))
			}
			if len(rule2.Ports) != 0 {
				t.Errorf("Rule 2: Expected 0 ports (all ports), got %d", len(rule2.Ports))
			}

			// Check same-job pod selector
			sameJobPeer := rule2.From[0]
			if sameJobPeer.PodSelector == nil {
				t.Fatal("Rule 2: PodSelector is nil")
			}
			if sameJobPeer.PodSelector.MatchLabels["jobset.sigs.k8s.io/jobset-name"] != tt.trainJob.Name {
				t.Errorf("Rule 2: Same-job selector = %q, want %q",
					sameJobPeer.PodSelector.MatchLabels["jobset.sigs.k8s.io/jobset-name"], tt.trainJob.Name)
			}
			// Should NOT have namespace selector (same namespace only)
			if sameJobPeer.NamespaceSelector != nil {
				t.Error("Rule 2: Should not have NamespaceSelector (same namespace)")
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
			name: "creates NetworkPolicy when progression tracking enabled",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "new-job",
					Namespace: "default",
					UID:       types.UID("uid-new"),
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "true",
					},
				},
			},
			existingPolicy:    nil,
			wantPolicyCreated: true,
			wantPolicyUpdated: false,
			wantErr:           false,
		},
		{
			name: "does nothing when progression tracking disabled",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "disabled-job",
					Namespace: "default",
					UID:       types.UID("uid-disabled"),
					Annotations: map[string]string{
						constants.AnnotationProgressionTracking: "false",
					},
				},
			},
			existingPolicy:    nil,
			wantPolicyCreated: false,
			wantPolicyUpdated: false,
			wantErr:           false,
		},
		{
			name: "does nothing when no annotations",
			trainJob: &trainer.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "no-annotation-job",
					Namespace: "default",
					UID:       types.UID("uid-no-ann"),
				},
			},
			existingPolicy:    nil,
			wantPolicyCreated: false,
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
					Name:      "existing-job" + constants.NetworkPolicyNameSuffix,
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
				// Verify the spec was updated (check port)
				if len(policy.Spec.Ingress) > 0 && len(policy.Spec.Ingress[0].Ports) > 0 {
					expectedPort := intstr.FromInt(9090)
					if *policy.Spec.Ingress[0].Ports[0].Port != expectedPort {
						t.Errorf("Expected port 9090 after update, got %v", policy.Spec.Ingress[0].Ports[0].Port)
					}
				}
			}

			if !tt.wantPolicyCreated && !tt.wantPolicyUpdated && tt.existingPolicy == nil {
				// Should not exist
				if getErr == nil {
					t.Error("Expected NetworkPolicy to NOT exist, but it was found")
				}
			}
		})
	}
}

func TestBuildNetworkPolicy_SecurityProperties(t *testing.T) {
	// This test verifies the security properties of the NetworkPolicy
	trainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "security-test",
			Namespace: "user-namespace",
			UID:       types.UID("security-uid"),
			Annotations: map[string]string{
				constants.AnnotationMetricsPort: "28080",
			},
		},
	}

	policy := buildNetworkPolicy(trainJob)

	t.Run("only allows Ingress policy type", func(t *testing.T) {
		if len(policy.Spec.PolicyTypes) != 1 {
			t.Fatalf("Expected 1 PolicyType, got %d", len(policy.Spec.PolicyTypes))
		}
		if policy.Spec.PolicyTypes[0] != networkingv1.PolicyTypeIngress {
			t.Errorf("PolicyType = %v, want Ingress", policy.Spec.PolicyTypes[0])
		}
	})

	t.Run("metrics port only accessible by controller", func(t *testing.T) {
		rule1 := policy.Spec.Ingress[0]

		// Should have exactly 1 peer (controller)
		if len(rule1.From) != 1 {
			t.Fatalf("Expected 1 peer for metrics rule, got %d", len(rule1.From))
		}

		// Should restrict to specific port
		if len(rule1.Ports) != 1 {
			t.Fatalf("Expected 1 port restriction, got %d", len(rule1.Ports))
		}

		// Verify controller labels are required
		peer := rule1.From[0]
		if peer.PodSelector.MatchLabels["app.kubernetes.io/name"] != "trainer" {
			t.Error("Missing trainer name label requirement")
		}
		if peer.PodSelector.MatchLabels["app.kubernetes.io/component"] != "controller" {
			t.Error("Missing manager component label requirement")
		}
	})

	t.Run("same-job pods cannot be spoofed from other namespaces", func(t *testing.T) {
		rule2 := policy.Spec.Ingress[1]

		// Should NOT have NamespaceSelector (restricts to same namespace)
		peer := rule2.From[0]
		if peer.NamespaceSelector != nil {
			t.Error("Same-job rule should NOT have NamespaceSelector (must be same namespace)")
		}
	})

	t.Run("other jobs in namespace cannot access metrics", func(t *testing.T) {
		rule2 := policy.Spec.Ingress[1]

		// Same-job selector should match ONLY this job's pods
		peer := rule2.From[0]
		if peer.PodSelector.MatchLabels["jobset.sigs.k8s.io/jobset-name"] != trainJob.Name {
			t.Error("Same-job rule should only match pods from this specific TrainJob")
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
		if ref.BlockOwnerDeletion == nil || !*ref.BlockOwnerDeletion {
			t.Error("OwnerReference should block owner deletion")
		}
	})
}
