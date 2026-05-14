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
	"testing"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
)

func TestGetManagementState(t *testing.T) {
	tests := []struct {
		name     string
		trainer  *Trainer
		expected common.ManagementState
	}{
		{
			name: "defaults to Managed when unset",
			trainer: &Trainer{
				Spec: TrainerSpec{
					ManagementState: "",
				},
			},
			expected: common.Managed,
		},
		{
			name: "returns Managed when explicitly set",
			trainer: &Trainer{
				Spec: TrainerSpec{
					ManagementState: common.Managed,
				},
			},
			expected: common.Managed,
		},
		{
			name: "returns Removed when set",
			trainer: &Trainer{
				Spec: TrainerSpec{
					ManagementState: common.Removed,
				},
			},
			expected: common.Removed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.trainer.GetManagementState()
			if got != tt.expected {
				t.Errorf("GetManagementState() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPlatformObjectInterface(t *testing.T) {
	trainer := &Trainer{}

	var _ common.PlatformObject = trainer

	if trainer.GetStatus() == nil {
		t.Error("GetStatus() returned nil")
	}

	conditions := []common.Condition{
		{
			Type:   "Ready",
			Status: "True",
		},
	}
	trainer.SetConditions(conditions)
	got := trainer.GetConditions()
	if len(got) != 1 || got[0].Type != "Ready" {
		t.Errorf("SetConditions/GetConditions failed: got %v, want %v", got, conditions)
	}

	if trainer.GetReleaseStatus() == nil {
		t.Error("GetReleaseStatus() returned nil")
	}

	releaseStatus := common.ComponentReleaseStatus{
		Releases: []common.ComponentRelease{
			{
				Name:    "trainer",
				Version: "v2.1.0",
			},
		},
	}
	trainer.SetReleaseStatus(releaseStatus)
	gotRelease := trainer.GetReleaseStatus()
	if len(gotRelease.Releases) != 1 || gotRelease.Releases[0].Name != "trainer" {
		t.Errorf("SetReleaseStatus/GetReleaseStatus failed: got %v, want %v", gotRelease, releaseStatus)
	}
}
