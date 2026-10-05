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
	"fmt"
	"os"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	applymetav1 "k8s.io/client-go/applyconfigurations/meta/v1"
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

// buildNetworkPolicyApplyConfiguration creates a NetworkPolicy apply configuration for the TrainJob's pods.
// Rule 1 (same-job pods → all ports) is always added for pod isolation.
// Rule 2 (controller → metrics port) is only added when progression tracking is enabled.
func buildNetworkPolicyApplyConfiguration(trainJob *trainer.TrainJob) *networkingv1apply.NetworkPolicyApplyConfiguration {
	ingressRules := []*networkingv1apply.NetworkPolicyIngressRuleApplyConfiguration{
		networkingv1apply.NetworkPolicyIngressRule().WithFrom(
			networkingv1apply.NetworkPolicyPeer().WithPodSelector(
				applymetav1.LabelSelector().WithMatchLabels(map[string]string{
					"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
				}),
			),
		),
	}

	// Add the controller → metrics-port rule only when progression tracking is enabled.
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

		ingressRules = append(ingressRules, networkingv1apply.NetworkPolicyIngressRule().
			WithFrom(networkingv1apply.NetworkPolicyPeer().
				WithNamespaceSelector(applymetav1.LabelSelector().WithMatchLabels(map[string]string{
					"kubernetes.io/metadata.name": controllerNamespace,
				})).
				WithPodSelector(applymetav1.LabelSelector().WithMatchLabels(map[string]string{
					constants.ControllerPodLabelName:      constants.ControllerPodLabelNameValue,
					constants.ControllerPodLabelComponent: constants.ControllerPodLabelComponentValue,
				})),
			).
			WithPorts(networkingv1apply.NetworkPolicyPort().
				WithProtocol(corev1.ProtocolTCP).
				WithPort(port),
			),
		)
	}

	return networkingv1apply.NetworkPolicy(getNetworkPolicyName(trainJob), trainJob.Namespace).
		WithLabels(map[string]string{
			"trainer.kubeflow.org/trainjob-name": trainJob.Name,
			"trainer.kubeflow.org/component":     "network-policy",
		}).
		WithOwnerReferences(applymetav1.OwnerReference().
			WithAPIVersion(trainer.SchemeGroupVersion.String()).
			WithKind("TrainJob").
			WithName(trainJob.Name).
			WithUID(trainJob.UID).
			WithController(true).
			WithBlockOwnerDeletion(true),
		).
		WithSpec(networkingv1apply.NetworkPolicySpec().
			WithPodSelector(applymetav1.LabelSelector().WithMatchLabels(map[string]string{
				"jobset.sigs.k8s.io/jobset-name": trainJob.Name,
			})).
			WithPolicyTypes(networkingv1.PolicyTypeIngress).
			WithIngress(ingressRules...),
		)
}

// ReconcileNetworkPolicy creates/updates NetworkPolicy for the TrainJob.
// Uses OwnerReference for automatic cleanup.
func ReconcileNetworkPolicy(ctx context.Context, c client.Client, trainJob *trainer.TrainJob) error {
	applyConfiguration := buildNetworkPolicyApplyConfiguration(trainJob)
	if err := c.Apply(ctx, applyConfiguration, client.FieldOwner("trainer"), client.ForceOwnership); err != nil {
		return fmt.Errorf("failed to apply NetworkPolicy: %w", err)
	}

	return nil
}
