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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/opendatahub-io/odh-platform-utilities/pkg/cluster"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/cluster/olm"
)

const (
	jobSetCRDName            = "jobsets.jobset.x-k8s.io"
	jobSetVersion            = "v1alpha2"
	jobSetGroup              = "jobset.x-k8s.io"
	jobSetKind               = "JobSet"
	jobSetOperatorCRName     = "cluster"
	jobSetOperatorName       = "jobset-operator"
	jobSetOperatorGroup      = "operator.openshift.io"
	jobSetOperatorKind       = "JobSetOperator"
	jobSetOperatorAPIVersion = "v1"
)

// checkJobSetAvailable verifies that the JobSet CRD exists and is established.
func (r *TrainerReconciler) checkJobSetAvailable(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	jobSetGK := schema.GroupKind{
		Group: jobSetGroup,
		Kind:  jobSetKind,
	}

	if err := cluster.CustomResourceDefinitionExists(ctx, r.Client, jobSetGK); err != nil {
		log.Error(err, "JobSet CRD not available", "crd", jobSetCRDName, "version", jobSetVersion)
		return false
	}

	log.V(1).Info("JobSet CRD is established", "name", jobSetCRDName)
	return true
}

func (r *TrainerReconciler) checkJobSetOperatorInstalled(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	operatorInfo, err := olm.OperatorExists(ctx, r.Client, jobSetOperatorName)
	if err != nil {
		log.Error(err, "Failed to verify JobSet Operator installation")
		return false
	}

	log.V(1).Info("JobSet Operator found via OLM", "version", operatorInfo.Version)
	return true
}

// checkJobSetOperatorCR checks that JobSetOperator CR exists with name "cluster".
func (r *TrainerReconciler) checkJobSetOperatorCR(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	jobSetOperatorCR := &unstructured.Unstructured{}
	jobSetOperatorCR.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   jobSetOperatorGroup,
		Version: jobSetOperatorAPIVersion,
		Kind:    jobSetOperatorKind,
	})

	if err := cluster.GetSingleton(ctx, r.Client, jobSetOperatorCR); err != nil {
		log.Error(err, "Failed to verify JobSetOperator CR", "expectedName", jobSetOperatorCRName)
		return false
	}

	log.V(1).Info("JobSetOperator CR found", "name", jobSetOperatorCRName)
	return true
}

func getJobSetOperatorNotInstalledMessage() string {
	return "JobSet Operator is not installed. " +
		"Please install the JobSet Operator via OLM (OperatorHub) before deploying Trainer."
}

func getJobSetOperatorCRMissingMessage() string {
	return fmt.Sprintf("JobSetOperator CR with name '%s' not found. "+
		"Please create the JobSetOperator CR to enable the JobSet controller.",
		jobSetOperatorCRName)
}

func getJobSetMissingMessage() string {
	return fmt.Sprintf("JobSet CRD (%s version %s) is required but not found. "+
		"Please install JobSet CRD before deploying Trainer.",
		jobSetCRDName, jobSetVersion)
}

func getJobSetMissingMessageOpenShift() string {
	return fmt.Sprintf("JobSet CRD (%s version %s) is required but not found. "+
		"This CRD should be created by the JobSet Operator. "+
		"Please check the JobSet Operator status or logs for more details.",
		jobSetCRDName, jobSetVersion)
}
