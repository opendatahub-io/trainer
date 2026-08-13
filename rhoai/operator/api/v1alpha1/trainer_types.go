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
	fwapi "github.com/opendatahub-io/odh-platform-utilities/framework/api"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ControllerResources struct {
	Name      string                      `json:"name"`
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
}

type TrainerSpec struct {
	AppNamespace string `json:"appNamespace,omitempty"`

	// +optional
	FeatureGates map[string]bool `json:"featureGates,omitempty"`

	// +optional
	Controllers []ControllerResources `json:"controllers,omitempty"`
}

type TrainerStatus struct {
	fwapi.Status                 `json:",inline"`
	fwapi.ComponentReleaseStatus `json:",inline"`
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

var _ fwapi.PlatformObject = &Trainer{}

func (t *Trainer) GetStatus() *fwapi.Status {
	return &t.Status.Status
}

func (t *Trainer) GetConditions() []fwapi.Condition {
	return t.Status.Conditions
}

func (t *Trainer) SetConditions(conditions []fwapi.Condition) {
	t.Status.Conditions = append(t.Status.Conditions[:0:0], conditions...)
}

func (t *Trainer) GetReleaseStatus() *[]fwapi.ComponentRelease {
	return &t.Status.Releases
}

func (t *Trainer) SetReleaseStatus(status []fwapi.ComponentRelease) {
	t.Status.Releases = append(t.Status.Releases[:0:0], status...)
}
