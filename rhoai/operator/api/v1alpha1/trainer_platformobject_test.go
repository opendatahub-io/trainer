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
	fwapi "github.com/opendatahub-io/odh-platform-utilities/framework/api"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestTrainerCommonTypeConversion(t *testing.T) {
	trainer := &Trainer{
		ObjectMeta: metav1.ObjectMeta{Name: "default-trainer"},
	}

	trainer.SetConditions([]fwapi.Condition{
		{Type: string(common.ConditionTypeReady), Status: metav1.ConditionTrue},
	})

	if len(trainer.GetConditions()) != 1 || trainer.GetConditions()[0].Type != string(common.ConditionTypeReady) {
		t.Fatalf("condition round-trip failed: %#v", trainer.GetConditions())
	}

	trainer.SetReleaseStatus([]fwapi.ComponentRelease{{Name: testComponentName, Version: "v2.1.0"}})
	releases := trainer.GetReleaseStatus()
	if len(*releases) != 1 || (*releases)[0].Name != testComponentName {
		t.Fatalf("release round-trip failed: %#v", *releases)
	}
}
