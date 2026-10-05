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
	"fmt"
	"os"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	networkingv1apply "k8s.io/client-go/applyconfigurations/networking/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/rhai/constants"
	"github.com/kubeflow/trainer/v2/pkg/rhai/progression"
)

const serviceAccountNamespaceFile = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

// getControllerNamespace returns the controller's namespace from SA mount.
func getControllerNamespace() string {
	if data, err := os.ReadFile(serviceAccountNamespaceFile); err == nil {
		if ns := strings.TrimSpace(string(data)); ns != "" {
			return ns
		}
	}
	return constants.DefaultControllerNamespace
}

func getNetworkPolicyName(trainJob *trainer.TrainJob) string {
	return trainJob.Name
}

// buildNetworkPolicy creates a NetworkPolicy for the TrainJob's pods.
// Rule 1 (same-job pods → all ports) is always added for pod isolation.
// Rule 2 (controller → metrics port) is only added when progression tracking is enabled.
func buildNetworkPolicy(trainJob *trainer.TrainJob) *networkingv1.NetworkPolicy {
	ingressRules := []networkingv1.NetworkPolicyIngressRule{}

	// Rule 1: Same-job pods → all ports (always, for NCCL/MPI/gRPC)
	ingressRules = append(ingressRules, networkingv1.NetworkPolicyIngressRule{
		From: []networkingv1.NetworkPolicyPeer{
			{
				PodSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
					},
				},
			},
		},
	})

	// Rule 2: Controller → metrics port (only when progression tracking enabled)
	if progression.IsProgressionTrackingEnabled(trainJob) {
		metricsPort := progression.GetMetricsPort(trainJob)
		portNum, err := strconv.Atoi(metricsPort)
		if err != nil {
			klog.Warningf("Invalid metrics port %q for TrainJob %s/%s, falling back to default %s",
				metricsPort, trainJob.Namespace, trainJob.Name, constants.DefaultMetricsPort)
			portNum, _ = strconv.Atoi(constants.DefaultMetricsPort)
		}
		port := intstr.FromInt(portNum)
		controllerNamespace := getControllerNamespace()

		ingressRules = append(ingressRules, networkingv1.NetworkPolicyIngressRule{
			From: []networkingv1.NetworkPolicyPeer{
				{
					NamespaceSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"kubernetes.io/metadata.name": controllerNamespace,
						},
					},
					PodSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							constants.ControllerPodLabelName:      constants.ControllerPodLabelNameValue,
							constants.ControllerPodLabelComponent: constants.ControllerPodLabelComponentValue,
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
		})
	}

	return &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getNetworkPolicyName(trainJob),
			Namespace: trainJob.Namespace,
			Labels: map[string]string{
				"trainer.kubeflow.org/trainjob-name": trainJob.Name,
				"trainer.kubeflow.org/component":     "network-policy",
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
			Ingress: ingressRules,
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func protocolPtr(p corev1.Protocol) *corev1.Protocol {
	return &p
}

func buildNetworkPolicyApplyConfiguration(trainJob *trainer.TrainJob) (*networkingv1apply.NetworkPolicyApplyConfiguration, error) {
	policy := buildNetworkPolicy(trainJob)
	policy.TypeMeta = metav1.TypeMeta{
		APIVersion: networkingv1.SchemeGroupVersion.String(),
		Kind:       "NetworkPolicy",
	}
	data, err := json.Marshal(policy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal NetworkPolicy: %w", err)
	}
	applyConfiguration := &networkingv1apply.NetworkPolicyApplyConfiguration{}
	if err := json.Unmarshal(data, applyConfiguration); err != nil {
		return nil, fmt.Errorf("failed to convert NetworkPolicy to apply configuration: %w", err)
	}
	return applyConfiguration, nil
}

// ReconcileNetworkPolicy creates/updates NetworkPolicy for the TrainJob.
// Uses OwnerReference for automatic cleanup.
func ReconcileNetworkPolicy(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) error {
	applyConfiguration, err := buildNetworkPolicyApplyConfiguration(trainJob)
	if err != nil {
		return err
	}
	if err := c.Apply(ctx, applyConfiguration, client.FieldOwner("trainer"), client.ForceOwnership); err != nil {
		return fmt.Errorf("failed to apply NetworkPolicy: %w", err)
	}

	return nil
}
