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

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	jobSetCRDName      = "jobsets.jobset.x-k8s.io"
	jobSetVersion      = "v1alpha2"
	jobSetResourceName = "jobsets"
)

// checkJobSetAvailable checks if the JobSet Operator is available by verifying
// the JobSet CRD exists and is established.
func (r *TrainerReconciler) checkJobSetAvailable(ctx context.Context) bool {
	log := logf.FromContext(ctx)

	// Query the API server for the JobSet CRD
	_, apiLists, err := r.DiscoveryClient.ServerGroupsAndResources()
	if err != nil {
		// Partial errors are acceptable - some API groups might be temporarily unavailable
		if !meta.IsNoMatchError(err) {
			log.V(1).Info("Discovery client returned partial error, continuing", "error", err)
		}
	}

	// Search for JobSet API group and version
	for _, apiList := range apiLists {
		if apiList.GroupVersion == fmt.Sprintf("jobset.x-k8s.io/%s", jobSetVersion) {
			for _, resource := range apiList.APIResources {
				if resource.Name == jobSetResourceName {
					log.V(1).Info("JobSet CRD found", "version", jobSetVersion)
					return true
				}
			}
		}
	}

	// Fallback: Try to get the CRD directly
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobSetCRDName,
		},
	}

	if err := r.Get(ctx, client.ObjectKeyFromObject(crd), crd); err == nil {
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

	log.Info("JobSet Operator not available", "crd", jobSetCRDName, "version", jobSetVersion)
	return false
}

// getJobSetMissingMessage returns a user-friendly message when JobSet is missing.
func getJobSetMissingMessage() string {
	return fmt.Sprintf("JobSet Operator (CRD %s version %s) is required but not found. "+
		"Please install the JobSet Operator before deploying Trainer. "+
		"TrainJob resources cannot be created without this dependency.",
		jobSetCRDName, jobSetVersion)
}
