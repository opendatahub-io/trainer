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

package e2e

import (
	"testing"

	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

func TestControllerPodRunning(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	var podName string
	verifyControllerUp := func(g Gomega) {
		pods, err := k8sClient.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "control-plane=controller-manager",
		})
		g.Expect(err).NotTo(HaveOccurred(), "Failed to retrieve controller-manager pod information")

		var activePods []corev1.Pod
		for _, p := range pods.Items {
			if p.DeletionTimestamp == nil {
				activePods = append(activePods, p)
			}
		}
		g.Expect(activePods).To(HaveLen(1), "expected 1 controller pod running")
		podName = activePods[0].Name
		g.Expect(podName).To(ContainSubstring("controller-manager"))
		g.Expect(string(activePods[0].Status.Phase)).To(Equal("Running"), "Incorrect controller-manager pod status")
	}
	g.Eventually(verifyControllerUp).Should(Succeed())
}

const (
	trainerNamespace = "opendatahub"

	jobSetCRDName      = "jobsets.jobset.x-k8s.io"
	jobSetVersion      = "v1alpha2"
	jobSetResourceName = "jobsets"
	jobSetOperatorName = "jobset-operator"
	jobSetSystemNs     = "jobset-system"
)

func TestTrainerReconciliation(t *testing.T) {
	g := NewWithT(t)
	k8sClient.RegisterDebugCleanup(t, ctx, namespace)

	// Create JobSet CRD to satisfy dependency check
	jobSetCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "jobset.x-k8s.io",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:   "JobSet",
				Plural: jobSetResourceName,
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    jobSetVersion,
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
		},
	}

	crdClient := k8sClient.ApiextensionsClient.ApiextensionsV1().CustomResourceDefinitions()
	_, err := crdClient.Create(ctx, jobSetCRD, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create JobSet CRD")
	t.Cleanup(func() {
		_ = crdClient.Delete(ctx, jobSetCRDName, metav1.DeleteOptions{})
	})

	// Wait for CRD to be established
	verifyJobSetCRDEstablished := func(g Gomega) {
		crd, err := crdClient.Get(ctx, jobSetCRDName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		established := false
		for _, cond := range crd.Status.Conditions {
			if cond.Type == apiextensionsv1.Established && cond.Status == apiextensionsv1.ConditionTrue {
				established = true
				break
			}
		}
		g.Expect(established).To(BeTrue(), "JobSet CRD should be established")
	}
	g.Eventually(verifyJobSetCRDEstablished).Should(Succeed())

	// Create jobset-system namespace
	jobSetNs := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetSystemNs,
		},
	}
	_, err = k8sClient.CoreV1().Namespaces().Create(ctx, jobSetNs, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create jobset-system namespace")
	t.Cleanup(func() {
		_ = k8sClient.CoreV1().Namespaces().Delete(ctx, jobSetSystemNs, metav1.DeleteOptions{})
	})

	// Create fake JobSet operator deployment to satisfy operator installation check
	jobSetOperatorDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobSetOperatorName + "-controller-manager",
			Namespace: jobSetSystemNs,
			Labels: map[string]string{
				"app.kubernetes.io/name": jobSetOperatorName,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: int32Ptr(1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": jobSetOperatorName,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": jobSetOperatorName,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "manager",
							Image: jobSetOperatorName + ":latest",
						},
					},
				},
			},
		},
	}
	_, err = k8sClient.AppsV1().Deployments(jobSetSystemNs).Create(ctx, jobSetOperatorDeployment, metav1.CreateOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create JobSet operator deployment")
	t.Cleanup(func() {
		deploymentName := jobSetOperatorName + "-controller-manager"
		_ = k8sClient.AppsV1().Deployments(jobSetSystemNs).Delete(ctx, deploymentName, metav1.DeleteOptions{})
	})

	err = k8sClient.CreateTrainer(ctx, common.Managed, trainerNamespace)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to create Trainer CR")
	t.Cleanup(func() {
		_ = k8sClient.DeleteTrainer(ctx)
	})

	verifyTrainerReconciled := func(g Gomega) {
		trainer, err := k8sClient.GetTrainer(ctx)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(trainer.Status.ObservedGeneration).To(Equal(trainer.Generation))
	}
	g.Eventually(verifyTrainerReconciled).Should(Succeed())

	ns, err := k8sClient.CoreV1().Namespaces().Get(ctx, trainerNamespace, metav1.GetOptions{})
	g.Expect(err).NotTo(HaveOccurred(), "Trainer namespace should exist")
	g.Expect(ns.Labels).To(HaveKeyWithValue("platform.opendatahub.io/part-of", "trainer"))

	deployments, err := k8sClient.AppsV1().Deployments(trainerNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: "platform.opendatahub.io/part-of=trainer",
	})
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(deployments.Items).NotTo(BeEmpty(), "Expected at least one Trainer deployment")

	err = k8sClient.DeleteTrainer(ctx)
	g.Expect(err).NotTo(HaveOccurred(), "Failed to delete Trainer CR")

	verifyTrainerDeleted := func(g Gomega) {
		_, err := k8sClient.GetTrainer(ctx)
		g.Expect(errors.IsNotFound(err)).To(BeTrue(), "Trainer CR should be deleted")
	}
	g.Eventually(verifyTrainerDeleted).Should(Succeed())
}

func int32Ptr(i int32) *int32 {
	return &i
}

// +kubebuilder:scaffold:e2e-webhooks-checks
