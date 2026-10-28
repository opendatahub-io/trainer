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
	frameworktls "github.com/opendatahub-io/odh-platform-utilities/framework/tls"
	configv1 "github.com/openshift/api/config/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const testAPIServerName = "cluster"
const testOpenSSLCipher = "ECDHE-RSA-AES128-GCM-SHA256"
const testUnsupportedMappedCipher = "ECDHE-RSA-AES256-SHA384"

func testScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = configv1.Install(s)
	return s
}

func testAPIServerClient(profile *configv1.TLSSecurityProfile, policy configv1.TLSAdherencePolicy) client.Client {
	return fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(
		&configv1.APIServer{
			ObjectMeta: metav1.ObjectMeta{Name: testAPIServerName},
			Spec: configv1.APIServerSpec{
				TLSAdherence:       policy,
				TLSSecurityProfile: profile,
			},
		},
	).Build()
}

type getErrorClient struct {
	client.Client
	err error
}

func (c getErrorClient) Get(_ context.Context, _ client.ObjectKey, _ client.Object, _ ...client.GetOption) error {
	return c.err
}

func TestResolveProfiles(t *testing.T) {
	custom := &configv1.TLSSecurityProfile{
		Type: configv1.TLSProfileCustomType,
		Custom: &configv1.CustomTLSProfile{
			TLSProfileSpec: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS11,
				Ciphers:       []string{testOpenSSLCipher, testUnsupportedMappedCipher},
			},
		},
	}
	tests := []struct {
		name          string
		profile       *configv1.TLSSecurityProfile
		policy        configv1.TLSAdherencePolicy
		wantHonored   bool
		wantMin       uint16
		wantCipherNum int
		wantCiphers   []uint16
		wantError     string
	}{
		{
			name:          "Old honors TLS 1.0 and configured ciphers",
			profile:       &configv1.TLSSecurityProfile{Type: configv1.TLSProfileOldType},
			policy:        configv1.TLSAdherencePolicyStrictAllComponents,
			wantHonored:   true,
			wantMin:       tls.VersionTLS10,
			wantCipherNum: 18,
		},
		{
			name:          "Intermediate",
			profile:       &configv1.TLSSecurityProfile{Type: configv1.TLSProfileIntermediateType},
			policy:        configv1.TLSAdherencePolicyStrictAllComponents,
			wantHonored:   true,
			wantMin:       tls.VersionTLS12,
			wantCipherNum: 6,
		},
		{
			name:          "Modern omits configurable TLS 1.3 ciphers",
			profile:       &configv1.TLSSecurityProfile{Type: configv1.TLSProfileModernType},
			policy:        configv1.TLSAdherencePolicyStrictAllComponents,
			wantHonored:   true,
			wantMin:       tls.VersionTLS13,
			wantCipherNum: 0,
		},
		{
			name:          "Custom preserves TLS 1.1 and maps OpenSSL cipher name",
			profile:       custom,
			policy:        configv1.TLSAdherencePolicyStrictAllComponents,
			wantHonored:   true,
			wantMin:       tls.VersionTLS11,
			wantCipherNum: 1,
			wantCiphers:   []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		{
			name:          "nil profile resolves to Intermediate",
			policy:        configv1.TLSAdherencePolicyStrictAllComponents,
			wantHonored:   true,
			wantMin:       tls.VersionTLS12,
			wantCipherNum: 6,
		},
		{
			name:          "unknown type under NoOpinion uses Intermediate",
			profile:       &configv1.TLSSecurityProfile{Type: configv1.TLSProfileType("FutureProfile")},
			wantMin:       tls.VersionTLS12,
			wantCipherNum: 6,
		},
		{
			name:      "unknown type under Strict fails closed",
			profile:   &configv1.TLSSecurityProfile{Type: configv1.TLSProfileType("FutureProfile")},
			policy:    configv1.TLSAdherencePolicyStrictAllComponents,
			wantError: "unsupported TLS profile type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewWithT(t)
			result, err := Resolve(t.Context(), testAPIServerClient(tt.profile, tt.policy), logr.Discard())
			if tt.wantError != "" {
				g.Expect(err).To(MatchError(ContainSubstring(tt.wantError)))
				g.Expect(result).To(BeNil())
				return
			}

			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(result.ProfileHonored).To(Equal(tt.wantHonored))
			g.Expect(result.Adherence).To(Equal(tt.policy))
			g.Expect(result.TLSOpts).To(HaveLen(2))

			cfg := &tls.Config{}
			for _, configure := range result.TLSOpts {
				configure(cfg)
			}
			g.Expect(cfg.MinVersion).To(Equal(tt.wantMin))
			g.Expect(cfg.CipherSuites).To(HaveLen(tt.wantCipherNum))
			if tt.wantCiphers != nil {
				g.Expect(cfg.CipherSuites).To(Equal(tt.wantCiphers))
			}
			g.Expect(cfg.NextProtos).To(Equal([]string{"h2", "http/1.1"}))
		})
	}
}

func TestShouldHonorClusterTLSProfile(t *testing.T) {
	tests := []struct {
		name    string
		policy  configv1.TLSAdherencePolicy
		honored bool
	}{
		{name: "NoOpinion", policy: configv1.TLSAdherencePolicyNoOpinion},
		{name: "Legacy", policy: configv1.TLSAdherencePolicyLegacyAdheringComponentsOnly},
		{name: "Strict", policy: configv1.TLSAdherencePolicyStrictAllComponents, honored: true},
		{name: "unknown policy fails secure", policy: configv1.TLSAdherencePolicy("FuturePolicy"), honored: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			NewWithT(t).Expect(frameworktls.ShouldHonorClusterTLSProfile(tt.policy)).To(Equal(tt.honored))
		})
	}
}

func TestResolveFetchFallbacks(t *testing.T) {
	apiserverGR := schema.GroupResource{Group: "config.openshift.io", Resource: "apiservers"}
	tests := []struct {
		name      string
		err       error
		watchable bool
		wantErr   bool
	}{
		{
			name: "NotFound disables watcher",
			err:  apierrors.NewNotFound(apiserverGR, testAPIServerName),
		},
		{
			name: "NoMatch disables watcher",
			err:  &apimeta.NoKindMatchError{GroupKind: schema.GroupKind{Group: "config.openshift.io", Kind: "APIServer"}},
		},
		{
			name:      "transient error uses fallback and arms watcher",
			err:       apierrors.NewServiceUnavailable("temporarily unavailable"),
			watchable: true,
		},
		{
			name:    "Forbidden fails closed",
			err:     apierrors.NewForbidden(apiserverGR, testAPIServerName, errors.New("denied")),
			wantErr: true,
		},
		{
			name:      "deadline exceeded uses fallback and arms watcher",
			err:       context.DeadlineExceeded,
			watchable: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewWithT(t)
			k8sClient := getErrorClient{
				Client: fake.NewClientBuilder().WithScheme(testScheme()).Build(),
				err:    tt.err,
			}
			result, err := Resolve(t.Context(), k8sClient, logr.Discard())
			if tt.wantErr {
				g.Expect(err).To(HaveOccurred())
				g.Expect(result).To(BeNil())
				return
			}

			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(result.UsedFallback).To(BeTrue())
			g.Expect(result.Watchable).To(Equal(tt.watchable))
			g.Expect(result.Adherence).To(Equal(configv1.TLSAdherencePolicyNoOpinion))
			g.Expect(result.ProfileHonored).To(BeFalse())
		})
	}
}

func TestResolveRejectsUnusableStrictProfiles(t *testing.T) {
	tests := []struct {
		name    string
		profile *configv1.TLSSecurityProfile
		wantErr error
	}{
		{
			name: "all ciphers unsupported",
			profile: &configv1.TLSSecurityProfile{
				Type: configv1.TLSProfileCustomType,
				Custom: &configv1.CustomTLSProfile{TLSProfileSpec: configv1.TLSProfileSpec{
					MinTLSVersion: configv1.VersionTLS12,
					Ciphers:       []string{testUnsupportedMappedCipher},
				}},
			},
			wantErr: frameworktls.ErrTLSProfileHasNoSupportedCipherSuites,
		},
		{
			name: "all groups unsupported",
			profile: &configv1.TLSSecurityProfile{
				Type: configv1.TLSProfileCustomType,
				Custom: &configv1.CustomTLSProfile{TLSProfileSpec: configv1.TLSProfileSpec{
					MinTLSVersion: configv1.VersionTLS12,
					Ciphers:       []string{testOpenSSLCipher},
					Groups:        []configv1.TLSGroup{"not-a-group"},
				}},
			},
			wantErr: frameworktls.ErrTLSProfileHasNoSupportedGroups,
		},
		{
			name: "invalid minimum version does not panic",
			profile: &configv1.TLSSecurityProfile{
				Type: configv1.TLSProfileCustomType,
				Custom: &configv1.CustomTLSProfile{TLSProfileSpec: configv1.TLSProfileSpec{
					MinTLSVersion: configv1.TLSProtocolVersion("VersionTLS99"),
					Ciphers:       []string{testOpenSSLCipher},
				}},
			},
			wantErr: frameworktls.ErrUnsupportedTLSProfileVersion,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewWithT(t)
			result, err := Resolve(t.Context(), testAPIServerClient(tt.profile, configv1.TLSAdherencePolicyStrictAllComponents), logr.Discard())
			g.Expect(result).To(BeNil())
			g.Expect(err).To(MatchError(ContainSubstring(tt.wantErr.Error())))
		})
	}
}

func TestResolveMapsMLKEMGroups(t *testing.T) {
	g := NewWithT(t)
	profile := &configv1.TLSSecurityProfile{
		Type: configv1.TLSProfileCustomType,
		Custom: &configv1.CustomTLSProfile{TLSProfileSpec: configv1.TLSProfileSpec{
			MinTLSVersion: configv1.VersionTLS12,
			Ciphers:       []string{testOpenSSLCipher},
			Groups: []configv1.TLSGroup{
				configv1.TLSGroupX25519MLKEM768,
				configv1.TLSGroupSecP256r1MLKEM768,
			},
		}},
	}
	result, err := Resolve(t.Context(), testAPIServerClient(profile, configv1.TLSAdherencePolicyStrictAllComponents), logr.Discard())
	g.Expect(err).NotTo(HaveOccurred())

	cfg := &tls.Config{}
	for _, configure := range result.TLSOpts {
		configure(cfg)
	}
	g.Expect(cfg.CurvePreferences).To(Equal([]tls.CurveID{tls.X25519MLKEM768, tls.SecP256r1MLKEM768}))
}

func TestResolveMapsOpenSSLCipherName(t *testing.T) {
	g := NewWithT(t)
	profile := &configv1.TLSSecurityProfile{
		Type: configv1.TLSProfileCustomType,
		Custom: &configv1.CustomTLSProfile{TLSProfileSpec: configv1.TLSProfileSpec{
			MinTLSVersion: configv1.VersionTLS12,
			Ciphers:       []string{testOpenSSLCipher},
		}},
	}
	result, err := Resolve(t.Context(), testAPIServerClient(profile, configv1.TLSAdherencePolicyStrictAllComponents), logr.Discard())
	g.Expect(err).NotTo(HaveOccurred())

	cfg := &tls.Config{}
	for _, configure := range result.TLSOpts {
		configure(cfg)
	}
	g.Expect(cfg.CipherSuites).To(Equal([]uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}))
}

func TestSetupWatcherSkipsUnwatchableResult(t *testing.T) {
	NewWithT(t).Expect(SetupWatcher(nil, &Result{Watchable: false}, func() {}, logr.Discard())).NotTo(HaveOccurred())
	NewWithT(t).Expect(SetupWatcher(nil, nil, func() {}, logr.Discard())).NotTo(HaveOccurred())
}

func TestNewSecurityProfileWatcherInitialStateAndCallbacks(t *testing.T) {
	tests := []struct {
		name              string
		result            Result
		profileCallback   bool
		adherenceCallback bool
		initialAdherence  configv1.TLSAdherencePolicy
	}{
		{
			name: "honored profile watches profile and adherence",
			result: Result{
				ProfileHonored: true,
				Adherence:      configv1.TLSAdherencePolicyStrictAllComponents,
			},
			profileCallback:   true,
			adherenceCallback: true,
			initialAdherence:  configv1.TLSAdherencePolicyStrictAllComponents,
		},
		{
			name: "transient fallback watches profile recovery from explicit NoOpinion",
			result: Result{
				UsedFallback: true,
				Watchable:    true,
				Adherence:    configv1.TLSAdherencePolicyNoOpinion,
			},
			profileCallback:   true,
			adherenceCallback: true,
			initialAdherence:  configv1.TLSAdherencePolicyNoOpinion,
		},
		{
			name: "non-honored loaded profile only watches adherence",
			result: Result{
				Watchable: true,
				Adherence: configv1.TLSAdherencePolicyLegacyAdheringComponentsOnly,
			},
			adherenceCallback: true,
			initialAdherence:  configv1.TLSAdherencePolicyLegacyAdheringComponentsOnly,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			watcher := newSecurityProfileWatcher(nil, &tt.result, func() {}, logr.Discard())
			g := NewWithT(t)
			if tt.profileCallback {
				g.Expect(watcher.OnProfileChange).NotTo(BeNil())
			} else {
				g.Expect(watcher.OnProfileChange).To(BeNil())
			}
			if tt.adherenceCallback {
				g.Expect(watcher.OnAdherencePolicyChange).NotTo(BeNil())
			} else {
				g.Expect(watcher.OnAdherencePolicyChange).To(BeNil())
			}
			g.Expect(watcher.InitialTLSAdherencePolicy).To(Equal(tt.initialAdherence))
		})
	}
}
