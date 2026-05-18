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

package controller

import (
	"context"
	"fmt"
	"strings"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	jobSetCRDName          = "jobsets.jobset.x-k8s.io"
	jobSetVersion          = "v1alpha2"
	jobSetOperatorCRName   = "cluster"
	jobSetOperatorName     = "jobset-operator"
	jobSetOperatorCSVMatch = "jobset-operator"
)

// checkJobSetAvailable verifies that the JobSet CRD exists and is established.
func (r *TrainerReconciler) checkJobSetAvailable(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
	}

	if err := r.Get(ctx, client.ObjectKeyFromObject(crd), crd); err != nil {
		log.Info("JobSet Operator not available", "crd", jobSetCRDName, "version", jobSetVersion)
		return false
	}

	// Check if CRD is established (ready to use)
	for _, cond := range crd.Status.Conditions {
		if cond.Type == apiextensionsv1.Established && cond.Status == apiextensionsv1.ConditionTrue {
			log.V(1).Info("JobSet CRD is established", "name", jobSetCRDName)
			return true
		}
	}

	log.Info("JobSet CRD found but not established yet", "name", jobSetCRDName)
	return false
}

func (r *TrainerReconciler) checkJobSetOperatorInstalled(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	// Check if OLM is available by checking for CSV CRD
	csvCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "clusterserviceversions.operators.coreos.com",
		},
	}

	if err := r.Get(ctx, client.ObjectKeyFromObject(csvCRD), csvCRD); err == nil {
		// OLM is available, check for CSV
		if csvFound := r.checkJobSetOperatorCSV(ctx); csvFound {
			log.V(1).Info("JobSet Operator found via ClusterServiceVersion")
			return true
		}
	} else {
		log.V(1).Info("OLM not available, checking for operator deployment")
	}

	// Fallback: check for deployment
	deploymentList := &unstructured.UnstructuredList{}
	deploymentList.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "DeploymentList",
	})

	// List all deployments across all namespaces
	if err := r.List(ctx, deploymentList); err != nil {
		log.Error(err, "Failed to list deployments")
		return false
	}

	// Look for jobset-operator deployment
	for _, item := range deploymentList.Items {
		name := item.GetName()
		labels := item.GetLabels()

		if strings.Contains(name, jobSetOperatorName) {
			log.V(1).Info("JobSet Operator found via deployment", "name", name, "namespace", item.GetNamespace())
			return true
		}

		// Also check labels
		if appName, ok := labels["app.kubernetes.io/name"]; ok && strings.Contains(appName, jobSetOperatorName) {
			log.V(1).Info("JobSet Operator found via deployment labels", "name", name, "namespace", item.GetNamespace())
			return true
		}
	}

	log.Info("JobSet Operator installation not found")
	return false
}

func (r *TrainerReconciler) checkJobSetOperatorCSV(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	csvList := &unstructured.UnstructuredList{}
	csvList.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "operators.coreos.com",
		Version: "v1alpha1",
		Kind:    "ClusterServiceVersionList",
	})

	if err := r.List(ctx, csvList); err != nil {
		log.V(1).Info("Failed to list ClusterServiceVersions, OLM may not be available", "error", err)
		return false
	}

	// Look for jobset-operator CSV
	for _, item := range csvList.Items {
		name := item.GetName()
		if strings.Contains(name, jobSetOperatorCSVMatch) {
			// Check if CSV is in a successful phase
			phase, found, err := unstructured.NestedString(item.Object, "status", "phase")
			if err == nil && found {
				if phase == "Succeeded" {
					log.V(1).Info("JobSet Operator CSV found and active", "name", name, "namespace", item.GetNamespace())
					return true
				}
				log.Info("JobSet Operator CSV found but not in Succeeded phase", "name", name, "phase", phase)
			}
		}
	}

	return false
}

// checkJobSetOperatorCR checks that JobSetOperator CR exists with name "cluster".
func (r *TrainerReconciler) checkJobSetOperatorCR(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	// Check if the JobSetOperator CRD exists first
	jobSetOperatorCRD := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "jobsetoperators.operator.openshift.io",
		},
	}

	if err := r.Get(ctx, client.ObjectKeyFromObject(jobSetOperatorCRD), jobSetOperatorCRD); err != nil {
		log.V(1).Info("JobSetOperator CRD not found, skipping CR check")
		return true // Not in OpenShift, skip this check
	}

	// CRD exists, now check for the CR
	jobSetOperatorCR := &unstructured.Unstructured{}
	jobSetOperatorCR.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "operator.openshift.io",
		Version: "v1",
		Kind:    "JobSetOperator",
	})

	if err := r.Get(ctx, client.ObjectKey{Name: jobSetOperatorCRName}, jobSetOperatorCR); err != nil {
		if errors.IsNotFound(err) {
			log.Info("JobSetOperator CR not found", "name", jobSetOperatorCRName)
			return false
		}
		log.Error(err, "Failed to check JobSetOperator CR", "name", jobSetOperatorCRName)
		return false
	}

	log.V(1).Info("JobSetOperator CR found", "name", jobSetOperatorCRName)
	return true
}

func getJobSetOperatorNotInstalledMessage() string {
	return "JobSet Operator is not installed. " +
		"Please install the JobSet Operator via OLM (OperatorHub) or manually deploy it before deploying Trainer."
}

func getJobSetOperatorCRMissingMessage() string {
	return fmt.Sprintf("JobSetOperator CR with name '%s' not found. "+
		"Please create the JobSetOperator CR to enable the JobSet controller.",
		jobSetOperatorCRName)
}

func getJobSetMissingMessage() string {
	return fmt.Sprintf("JobSet CRD (%s version %s) is required but not found. "+
		"This CRD should be created by the JobSet Operator. "+
		"Please check the JobSet Operator status or logs for more details.",
		jobSetCRDName, jobSetVersion)
}
