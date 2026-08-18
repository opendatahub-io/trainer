/*
Copyright The Kubeflow Authors.

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

package tls

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic/fake"
	k8stesting "k8s.io/client-go/testing"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func newAPIServerObj(profile map[string]interface{}) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "config.openshift.io", Version: "v1", Kind: "APIServer",
	})
	obj.SetName("cluster")
	if profile != nil {
		_ = unstructured.SetNestedMap(obj.Object, profile, "spec", "tlsSecurityProfile")
	}
	return obj
}

func newWatcher(dynClient *fake.FakeDynamicClient, initial interface{}, cb func()) *ProfileWatcher {
	w := NewProfileWatcher(nil, initial, cb)
	w.dynClient = dynClient
	return w
}

func fakeDynClient(objs ...runtime.Object) *fake.FakeDynamicClient {
	scheme := runtime.NewScheme()
	return fake.NewSimpleDynamicClientWithCustomListKinds(scheme,
		map[schema.GroupVersionResource]string{
			apiServerGVR: "APIServerList",
		}, objs...)
}

func TestReconcile_ProfileUnchanged(t *testing.T) {
	initial := map[string]interface{}{"type": "Intermediate"}
	obj := newAPIServerObj(initial)

	called := false
	w := newWatcher(fakeDynClient(obj), initial, func() { called = true })

	result, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Errorf("expected no requeue, got %v", result.RequeueAfter)
	}
	if called {
		t.Error("onProfileChange must not be called when profile is unchanged")
	}
}

func TestReconcile_ProfileChanged(t *testing.T) {
	initial := map[string]interface{}{"type": "Intermediate"}
	updated := map[string]interface{}{"type": "Modern"}
	obj := newAPIServerObj(updated)

	called := false
	w := newWatcher(fakeDynClient(obj), initial, func() { called = true })

	result, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Errorf("expected no requeue, got %v", result.RequeueAfter)
	}
	if !called {
		t.Error("onProfileChange must be called when profile changes")
	}
	if w.lastProfile.(map[string]interface{})["type"] != "Modern" {
		t.Errorf("lastProfile not updated: %v", w.lastProfile)
	}
}

func TestReconcile_NilCallback(t *testing.T) {
	initial := map[string]interface{}{"type": "Intermediate"}
	updated := map[string]interface{}{"type": "Modern"}
	obj := newAPIServerObj(updated)

	// nil callback must not panic
	w := newWatcher(fakeDynClient(obj), initial, nil)
	if _, err := w.Reconcile(context.Background(), reconcile.Request{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestReconcile_APIServerNotFound(t *testing.T) {
	dynClient := fakeDynClient()
	dynClient.PrependReactor("get", "apiservers", func(_ k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.NewNotFound(apiServerGVR.GroupResource(), "cluster")
	})

	called := false
	w := newWatcher(dynClient, nil, func() { called = true })

	result, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != profileRetryInterval {
		t.Errorf("expected requeue after %v, got %v", profileRetryInterval, result.RequeueAfter)
	}
	if called {
		t.Error("onProfileChange must not be called on API error")
	}
}

func TestReconcile_APIServerTransientError(t *testing.T) {
	dynClient := fakeDynClient()
	dynClient.PrependReactor("get", "apiservers", func(_ k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.NewServiceUnavailable("temporarily unavailable")
	})

	called := false
	w := newWatcher(dynClient, nil, func() { called = true })

	result, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != profileRetryInterval {
		t.Errorf("expected requeue after %v, got %v", profileRetryInterval, result.RequeueAfter)
	}
	if called {
		t.Error("onProfileChange must not be called on transient error")
	}
}

func TestReconcile_NoProfile(t *testing.T) {
	obj := newAPIServerObj(nil)

	called := false
	w := newWatcher(fakeDynClient(obj), map[string]interface{}{"type": "Intermediate"}, func() { called = true })

	_, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("onProfileChange must be called when profile changes from non-nil to nil")
	}
}

// TestReconcile_InitialNilToProfile covers the first-time watcher sees a profile.
func TestReconcile_InitialNilToProfile(t *testing.T) {
	profile := map[string]interface{}{"type": "Old"}
	obj := newAPIServerObj(profile)

	called := false
	// Start with nil initial profile — any incoming profile counts as a change.
	w := newWatcher(fakeDynClient(obj), nil, func() { called = true })

	_, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("onProfileChange must be called when initial profile is nil and API returns a profile")
	}
}

// TestReconcile_DeepProfileChange covers a nested field change inside tlsSecurityProfile.
func TestReconcile_DeepProfileChange(t *testing.T) {
	initial := map[string]interface{}{
		"type": "Custom",
		"custom": map[string]interface{}{
			"ciphers": []interface{}{"ECDHE-RSA-AES128-GCM-SHA256"},
		},
	}
	updated := map[string]interface{}{
		"type": "Custom",
		"custom": map[string]interface{}{
			"ciphers": []interface{}{"ECDHE-RSA-AES256-GCM-SHA384"},
		},
	}
	obj := newAPIServerObj(updated)

	called := false
	w := newWatcher(fakeDynClient(obj), initial, func() { called = true })

	_, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("onProfileChange must be called when nested cipher list changes")
	}
}

// TestReconcile_ClientCreationFails covers the path where dynClient is nil and restConfig is also nil.
func TestReconcile_ClientCreationFails(t *testing.T) {
	called := false
	// dynClient nil + nil restConfig → dynamic.NewForConfig returns an error
	w := &ProfileWatcher{
		restConfig:      nil,
		dynClient:       nil,
		lastProfile:     nil,
		onProfileChange: func() { called = true },
	}

	result, err := w.Reconcile(context.Background(), reconcile.Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != profileRetryInterval {
		t.Errorf("expected requeue after %v on client creation failure, got %v", profileRetryInterval, result.RequeueAfter)
	}
	if called {
		t.Error("onProfileChange must not be called when client creation fails")
	}
}
