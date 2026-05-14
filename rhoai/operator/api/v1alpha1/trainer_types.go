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

package v1alpha1

import (
	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ControllerResources struct {
	Name      string                      `json:"name"`
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
}

type TrainerSpec struct {
	// +kubebuilder:validation:Enum=Managed;Removed
	// +kubebuilder:default=Managed
	ManagementState common.ManagementState `json:"managementState,omitempty"`

	AppNamespace string `json:"appNamespace,omitempty"`

	// +optional
	FeatureGates map[string]bool `json:"featureGates,omitempty"`

	// +optional
	Controllers []ControllerResources `json:"controllers,omitempty"`
}

type TrainerStatus struct {
	common.Status                 `json:",inline"`
	common.ComponentReleaseStatus `json:",inline"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'default-trainer'",message="Trainer must be named 'default-trainer'"
// +kubebuilder:printcolumn:name="Ready",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status"
// +kubebuilder:printcolumn:name="Reason",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].reason"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// Trainer is the Schema for the trainers API.
type Trainer struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrainerSpec   `json:"spec,omitempty"`
	Status TrainerStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TrainerList contains a list of Trainer.
type TrainerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Trainer `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Trainer{}, &TrainerList{})
}

var _ common.PlatformObject = &Trainer{}

func (t *Trainer) GetStatus() *common.Status {
	return &t.Status.Status
}

func (t *Trainer) GetConditions() []common.Condition {
	return t.Status.Conditions
}

func (t *Trainer) SetConditions(conditions []common.Condition) {
	t.Status.Conditions = conditions
}

func (t *Trainer) GetReleaseStatus() *common.ComponentReleaseStatus {
	return &t.Status.ComponentReleaseStatus
}

func (t *Trainer) SetReleaseStatus(status common.ComponentReleaseStatus) {
	t.Status.ComponentReleaseStatus = status
}

func (t *Trainer) GetManagementState() common.ManagementState {
	if t.Spec.ManagementState == "" {
		return common.Managed
	}
	return t.Spec.ManagementState
}
