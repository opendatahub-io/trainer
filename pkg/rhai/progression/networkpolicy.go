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
	"fmt"
	"os"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
)

const serviceAccountNamespaceFile = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

// getControllerNamespace returns the controller's namespace.
// Tries: SA namespace file > CONTROLLER_NAMESPACE env > default.
func getControllerNamespace() string {
	if data, err := os.ReadFile(serviceAccountNamespaceFile); err == nil {
		if ns := strings.TrimSpace(string(data)); ns != "" {
			return ns
		}
	}
	if ns := os.Getenv("CONTROLLER_NAMESPACE"); ns != "" {
		return ns
	}
	return constants.DefaultControllerNamespace
}

func getNetworkPolicyName(trainJob *trainer.TrainJob) string {
	return trainJob.Name + constants.NetworkPolicyNameSuffix
}

// buildNetworkPolicy creates a NetworkPolicy restricting metrics port access to controller only.
func buildNetworkPolicy(trainJob *trainer.TrainJob) *networkingv1.NetworkPolicy {
	metricsPort := GetMetricsPort(trainJob)
	portNum, err := strconv.Atoi(metricsPort)
	if err != nil {
		portNum = 28080 // default
	}
	port := intstr.FromInt(portNum)

	controllerNamespace := getControllerNamespace()

	return &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getNetworkPolicyName(trainJob),
			Namespace: trainJob.Namespace,
			Labels: map[string]string{
				"trainer.kubeflow.org/trainjob-name": trainJob.Name,
				"trainer.kubeflow.org/component":     "metrics-security",
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion:         trainer.SchemeGroupVersion.String(),
					Kind:               "TrainJob",
					Name:               trainJob.Name,
					UID:                trainJob.UID,
					Controller:         boolPtr(true),
					BlockOwnerDeletion: boolPtr(true),
				},
			},
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
				},
			},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					// Rule 1: Controller → metrics port
					From: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"kubernetes.io/metadata.name": controllerNamespace,
								},
							},
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"app.kubernetes.io/name":      "trainer",
									"app.kubernetes.io/component": "controller",
								},
							},
						},
					},
					Ports: []networkingv1.NetworkPolicyPort{
						{
							Protocol: protocolPtr(corev1.ProtocolTCP),
							Port:     &port,
						},
					},
				},
				{
					// Rule 2: Same-job pods → all ports (for NCCL/MPI/gRPC)
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
								},
							},
						},
					},
				},
			},
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func protocolPtr(p corev1.Protocol) *corev1.Protocol {
	return &p
}

// ReconcileNetworkPolicy creates/updates NetworkPolicy for metrics endpoint security.
// Uses OwnerReference for automatic cleanup.
func ReconcileNetworkPolicy(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) error {
	if !IsProgressionTrackingEnabled(trainJob) {
		return nil
	}

	desiredPolicy := buildNetworkPolicy(trainJob)
	existingPolicy := &networkingv1.NetworkPolicy{}
	err := c.Get(ctx, client.ObjectKey{
		Namespace: trainJob.Namespace,
		Name:      getNetworkPolicyName(trainJob),
	}, existingPolicy)

	if apierrors.IsNotFound(err) {
		if createErr := c.Create(ctx, desiredPolicy); createErr != nil {
			return fmt.Errorf("failed to create NetworkPolicy: %w", createErr)
		}
		return nil
	}

	if err != nil {
		return fmt.Errorf("failed to get NetworkPolicy: %w", err)
	}

	existingPolicy.Spec = desiredPolicy.Spec
	existingPolicy.Labels = desiredPolicy.Labels
	if updateErr := c.Update(ctx, existingPolicy); updateErr != nil {
		return fmt.Errorf("failed to update NetworkPolicy: %w", updateErr)
	}

	return nil
}
