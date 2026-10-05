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
	"encoding/json"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
)

type applyClient struct {
	client.Client
}

func (c *applyClient) Apply(ctx context.Context, obj runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	desired := &networkingv1.NetworkPolicy{}
	if err := json.Unmarshal(data, desired); err != nil {
		return err
	}
	existing := &networkingv1.NetworkPolicy{}
	err = c.Get(ctx, client.ObjectKeyFromObject(desired), existing)
	if apierrors.IsNotFound(err) {
		return c.Create(ctx, desired)
	}
	if err != nil {
		return err
	}
	existing.Spec = desired.Spec
	existing.Labels = desired.Labels
	existing.OwnerReferences = desired.OwnerReferences
	return c.Update(ctx, existing)
}

func TestReconcileNetworkPolicy(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := trainer.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := networkingv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name           string
		trainJob       *trainer.TrainJob
		existingPolicy *networkingv1.NetworkPolicy
		wantSpec       networkingv1.NetworkPolicySpec
	}{
		{
			name: "progression tracking disabled",
			trainJob: &trainer.TrainJob{ObjectMeta: metav1.ObjectMeta{
				Name:      "without-progression",
				Namespace: "default",
				UID:       types.UID("without-progression-uid"),
			}},
			wantSpec: networkingv1.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{MatchLabels: map[string]string{
					"jobset.sigs.k8s.io/jobset-name": "without-progression",
				}},
				PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
				Ingress: []networkingv1.NetworkPolicyIngressRule{{
					From: []networkingv1.NetworkPolicyPeer{{
						PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
							"jobset.sigs.k8s.io/jobset-name": "without-progression",
						}},
					}},
				}},
			},
		},
		{
			name: "progression tracking enabled",
			trainJob: &trainer.TrainJob{ObjectMeta: metav1.ObjectMeta{
				Name:      "with-progression",
				Namespace: "default",
				UID:       types.UID("with-progression-uid"),
				Annotations: map[string]string{
					constants.AnnotationProgressionTracking: "true",
					constants.AnnotationMetricsPort:         "9090",
				},
			}},
			existingPolicy: &networkingv1.NetworkPolicy{ObjectMeta: metav1.ObjectMeta{
				Name:      "with-progression",
				Namespace: "default",
			}},
			wantSpec: networkingv1.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{MatchLabels: map[string]string{
					"jobset.sigs.k8s.io/jobset-name": "with-progression",
				}},
				PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
				Ingress: []networkingv1.NetworkPolicyIngressRule{
					{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
								"jobset.sigs.k8s.io/jobset-name": "with-progression",
							}},
						}},
					},
					{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
								"kubernetes.io/metadata.name": constants.DefaultControllerNamespace,
							}},
							PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
								constants.ControllerPodLabelName:      constants.ControllerPodLabelNameValue,
								constants.ControllerPodLabelComponent: constants.ControllerPodLabelComponentValue,
							}},
						}},
						Ports: []networkingv1.NetworkPolicyPort{{
							Protocol: protocolPointer(corev1.ProtocolTCP),
							Port:     intOrStringPointer(intstr.FromInt(9090)),
						}},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clientBuilder := fake.NewClientBuilder().WithScheme(scheme)
			if tt.existingPolicy != nil {
				clientBuilder = clientBuilder.WithObjects(tt.existingPolicy)
			}
			c := &applyClient{Client: clientBuilder.Build()}

			if err := ReconcileNetworkPolicy(context.Background(), c, tt.trainJob); err != nil {
				t.Fatalf("ReconcileNetworkPolicy() error = %v", err)
			}

			actual := &networkingv1.NetworkPolicy{}
			if err := c.Get(context.Background(), client.ObjectKeyFromObject(tt.trainJob), actual); err != nil {
				t.Fatalf("failed to get NetworkPolicy: %v", err)
			}
			if !reflect.DeepEqual(actual.Spec, tt.wantSpec) {
				t.Errorf("NetworkPolicy spec = %#v, want %#v", actual.Spec, tt.wantSpec)
			}
		})
	}
}

func protocolPointer(protocol corev1.Protocol) *corev1.Protocol {
	return &protocol
}

func intOrStringPointer(value intstr.IntOrString) *intstr.IntOrString {
	return &value
}
