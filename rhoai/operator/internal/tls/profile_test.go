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

package tls

import (
	"context"
	"crypto/tls"
	"errors"
	"testing"

	"github.com/go-logr/logr"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/types"
	configv1 "github.com/openshift/api/config/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func testScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = configv1.Install(s)
	return s
}

type getErrorClient struct {
	client.Client
	err error
}

func (c getErrorClient) Get(_ context.Context, _ client.ObjectKey, _ client.Object, _ ...client.GetOption) error {
	return c.err
}

func TestResolve(t *testing.T) {
	intermediate := configv1.TLSProfiles[configv1.TLSProfileIntermediateType]
	modern := configv1.TLSProfiles[configv1.TLSProfileModernType]
	apiserverGR := schema.GroupResource{Group: "config.openshift.io", Resource: "apiservers"}

	tests := []struct {
		name           string
		client         client.Client
		err            types.GomegaMatcher
		profileFetched types.GomegaMatcher
		minVersion     types.GomegaMatcher
	}{
		{
			name:           "APIServer not found uses Intermediate and skips watcher",
			client:         fake.NewClientBuilder().WithScheme(testScheme()).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeFalse(),
			minVersion:     Equal(intermediate.MinTLSVersion),
		},
		{
			name: "APIServer with Modern profile is applied",
			client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
				&configv1.APIServer{
					ObjectMeta: metav1.ObjectMeta{Name: "cluster"},
					Spec: configv1.APIServerSpec{
						TLSSecurityProfile: &configv1.TLSSecurityProfile{
							Type: configv1.TLSProfileModernType,
						},
					},
				},
			).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeTrue(),
			minVersion:     Equal(modern.MinTLSVersion),
		},
		{
			name: "Forbidden fails closed",
			client: getErrorClient{
				Client: fake.NewClientBuilder().WithScheme(testScheme()).Build(),
				err:    apierrors.NewForbidden(apiserverGR, "cluster", errors.New("denied")),
			},
			err: MatchError(ContainSubstring("reading APIServer TLS profile")),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewWithT(t)

			result, err := Resolve(t.Context(), tt.client, logr.Discard())
			g.Expect(err).Should(tt.err)
			if tt.profileFetched == nil {
				g.Expect(result).Should(BeNil())
				return
			}
			g.Expect(result.ProfileFetched).Should(tt.profileFetched)
			g.Expect(result.Profile.MinTLSVersion).Should(tt.minVersion)
			g.Expect(result.TLSOpts).ShouldNot(BeEmpty())

			cfg := &tls.Config{}
			for _, fn := range result.TLSOpts {
				fn(cfg)
			}
			g.Expect(cfg.NextProtos).Should(Equal([]string{"h2", "http/1.1"}))
		})
	}
}

func TestSetupWatcherSkipsWhenProfileNotFetched(t *testing.T) {
	g := NewWithT(t)
	g.Expect(SetupWatcher(nil, &Result{ProfileFetched: false}, func() {}, logr.Discard())).
		ShouldNot(HaveOccurred())
}

func TestSetupWatcherSkipsNilResult(t *testing.T) {
	g := NewWithT(t)
	g.Expect(SetupWatcher(nil, nil, func() {}, logr.Discard())).ShouldNot(HaveOccurred())
}
