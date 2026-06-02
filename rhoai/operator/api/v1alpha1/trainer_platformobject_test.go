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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	"github.com/opendatahub-io/odh-platform-utilities/api/common/validation"
)

// TestTrainer_PlatformObjectContract verifies that Trainer correctly implements
// the PlatformObject behavioral contract, including:
// - Status pointer semantics (must return pointer to actual field, not copy)
// - Condition round-trip persistence
// - Mandatory conditions (Ready, ProvisioningSucceeded, Degraded)
// - Release status handling
//
// This test uses the validation helper from odh-platform-utilities to ensure
// compliance with the platform object contract.
func TestTrainer_PlatformObjectContract(t *testing.T) {
	obj := &Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default-trainer",
		},
		Spec: TrainerSpec{
			ManagementState: common.Managed,
		},
	}

	// Validate that Trainer satisfies the PlatformObject contract
	validation.ValidatePlatformObject(t, obj)
}
