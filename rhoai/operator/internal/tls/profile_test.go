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

const testAPIServerName = "cluster"

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

type secondGetErrorClient struct {
	client.Client
	err  error
	gets int
}

func (c *secondGetErrorClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if c.gets == 0 {
		c.gets++
		return c.Client.Get(ctx, key, obj, opts...)
	}
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
		profileHonored types.GomegaMatcher
		profileVersion types.GomegaMatcher
		appliedVersion uint16
	}{
		{
			name:           "APIServer not found uses Intermediate and skips watcher",
			client:         fake.NewClientBuilder().WithScheme(testScheme()).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeFalse(),
			profileHonored: BeFalse(),
			profileVersion: Equal(intermediate.MinTLSVersion),
			appliedVersion: tls.VersionTLS12,
		},
		{
			name: "APIServer with Modern profile is applied",
			client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
				&configv1.APIServer{
					ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
					Spec: configv1.APIServerSpec{
						TLSAdherence: configv1.TLSAdherencePolicyStrictAllComponents,
						TLSSecurityProfile: &configv1.TLSSecurityProfile{
							Type: configv1.TLSProfileModernType,
						},
					},
				},
			).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeTrue(),
			profileHonored: BeTrue(),
			profileVersion: Equal(modern.MinTLSVersion),
			appliedVersion: tls.VersionTLS13,
		},
		{
			name: "APIServer with unknown adherence honors the profile",
			client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
				&configv1.APIServer{
					ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
					Spec: configv1.APIServerSpec{
						TLSAdherence: configv1.TLSAdherencePolicy("FuturePolicy"),
						TLSSecurityProfile: &configv1.TLSSecurityProfile{
							Type: configv1.TLSProfileModernType,
						},
					},
				},
			).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeTrue(),
			profileHonored: BeTrue(),
			profileVersion: Equal(modern.MinTLSVersion),
			appliedVersion: tls.VersionTLS13,
		},
		{
			name: "APIServer with unset adherence uses Intermediate defaults",
			client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
				&configv1.APIServer{
					ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
					Spec: configv1.APIServerSpec{
						TLSSecurityProfile: &configv1.TLSSecurityProfile{
							Type: configv1.TLSProfileModernType,
						},
					},
				},
			).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeTrue(),
			profileHonored: BeFalse(),
			profileVersion: Equal(modern.MinTLSVersion),
			appliedVersion: tls.VersionTLS12,
		},
		{
			name: "APIServer with Legacy adherence uses Intermediate defaults",
			client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
				&configv1.APIServer{
					ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
					Spec: configv1.APIServerSpec{
						TLSAdherence: configv1.TLSAdherencePolicyLegacyAdheringComponentsOnly,
						TLSSecurityProfile: &configv1.TLSSecurityProfile{
							Type: configv1.TLSProfileModernType,
						},
					},
				},
			).Build(),
			err:            Not(HaveOccurred()),
			profileFetched: BeTrue(),
			profileHonored: BeFalse(),
			profileVersion: Equal(modern.MinTLSVersion),
			appliedVersion: tls.VersionTLS12,
		},
		{
			name: "Forbidden fails closed",
			client: getErrorClient{
				Client: fake.NewClientBuilder().WithScheme(testScheme()).Build(),
				err:    apierrors.NewForbidden(apiserverGR, testAPIServerName, errors.New("denied")),
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
			g.Expect(result.ProfileHonored).Should(tt.profileHonored)
			g.Expect(result.Profile.MinTLSVersion).Should(tt.profileVersion)
			g.Expect(result.TLSOpts).ShouldNot(BeEmpty())

			cfg := &tls.Config{}
			for _, fn := range result.TLSOpts {
				fn(cfg)
			}
			g.Expect(cfg.MinVersion).Should(Equal(tt.appliedVersion))
			g.Expect(cfg.NextProtos).Should(Equal([]string{"h2", "http/1.1"}))
		})
	}
}

func TestResolveAppliesTLSGroups(t *testing.T) {
	g := NewWithT(t)
	k8sClient := fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
		&configv1.APIServer{
			ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
			Spec: configv1.APIServerSpec{
				TLSAdherence: configv1.TLSAdherencePolicyStrictAllComponents,
				TLSSecurityProfile: &configv1.TLSSecurityProfile{
					Type: configv1.TLSProfileCustomType,
					Custom: &configv1.CustomTLSProfile{
						TLSProfileSpec: configv1.TLSProfileSpec{
							Groups:        []configv1.TLSGroup{configv1.TLSGroupX25519, configv1.TLSGroupSecP256r1},
							MinTLSVersion: configv1.VersionTLS12,
						},
					},
				},
			},
		},
	).Build()

	result, err := Resolve(t.Context(), k8sClient, logr.Discard())
	g.Expect(err).NotTo(HaveOccurred())

	cfg := &tls.Config{}
	for _, fn := range result.TLSOpts {
		fn(cfg)
	}
	g.Expect(cfg.CurvePreferences).To(Equal([]tls.CurveID{tls.X25519, tls.CurveP256}))
}

func TestResolveFailsClosedWhenAdherenceReadIsTransient(t *testing.T) {
	g := NewWithT(t)
	k8sClient := &secondGetErrorClient{
		Client: fake.NewClientBuilder().WithScheme(testScheme()).WithRuntimeObjects(
			&configv1.APIServer{
				ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
				Spec: configv1.APIServerSpec{
					TLSAdherence: configv1.TLSAdherencePolicyStrictAllComponents,
					TLSSecurityProfile: &configv1.TLSSecurityProfile{
						Type: configv1.TLSProfileModernType,
					},
				},
			},
		).Build(),
		err: apierrors.NewServiceUnavailable("adherence API unavailable"),
	}

	result, err := Resolve(t.Context(), k8sClient, logr.Discard())
	g.Expect(result).To(BeNil())
	g.Expect(err).To(MatchError(ContainSubstring("reading APIServer TLS adherence policy")))
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

func TestNewSecurityProfileWatcherCallbacks(t *testing.T) {
	g := NewWithT(t)

	legacyWatcher := newSecurityProfileWatcher(nil, &Result{
		AdherenceFetched: true,
		ProfileHonored:   false,
	}, func() {}, logr.Discard())
	g.Expect(legacyWatcher.OnProfileChange).To(BeNil())
	g.Expect(legacyWatcher.OnAdherencePolicyChange).NotTo(BeNil())

	strictWatcher := newSecurityProfileWatcher(nil, &Result{
		AdherenceFetched: true,
		ProfileHonored:   true,
	}, func() {}, logr.Discard())
	g.Expect(strictWatcher.OnProfileChange).NotTo(BeNil())
	g.Expect(strictWatcher.OnAdherencePolicyChange).NotTo(BeNil())
}
