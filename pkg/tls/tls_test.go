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
	"crypto/tls"
	"slices"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	configv1 "github.com/openshift/api/config/v1"
	tlspkg "github.com/openshift/controller-runtime-common/pkg/tls"
)

func TestSetupWatcherSkipsUnavailableProfile(t *testing.T) {
	if err := SetupWatcher(nil, &Result{}, func() {}, logr.Discard()); err != nil {
		t.Fatalf("SetupWatcher() returned an error: %v", err)
	}
}

func TestValidateProfileSpecRejectsUnusableProfiles(t *testing.T) {
	cases := map[string]struct {
		profile    configv1.TLSProfileSpec
		wantErrSub string
	}{
		"unknown minimum version": {
			profile:    configv1.TLSProfileSpec{MinTLSVersion: "VersionTLS99"},
			wantErrSub: "VersionTLS99",
		},
		"all ciphers unusable": {
			profile: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS12,
				Ciphers:       []string{"not-a-cipher"},
			},
			wantErrSub: "not-a-cipher",
		},
		"TLS 1.3-only ciphers with TLS 1.2 minimum": {
			profile: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS12,
				Ciphers:       []string{"TLS_AES_128_GCM_SHA256"},
			},
			wantErrSub: "TLS_AES_128_GCM_SHA256",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := validateProfileSpec(logr.Discard(), tc.profile); err == nil {
				t.Fatal("Expected an error, got nil")
			} else if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Errorf("Expected error to mention %q, got: %v", tc.wantErrSub, err)
			}
		})
	}
}

// TestValidatedProfileAppliedToTLSConfig mirrors how Resolve assembles its
// TLSOpts (validation, then the shared profile callback and next-protos) and
// applies them to a fresh tls.Config, without needing a cluster client.
func TestValidatedProfileAppliedToTLSConfig(t *testing.T) {
	apply := func(t *testing.T, profile configv1.TLSProfileSpec) *tls.Config {
		t.Helper()
		validated, err := validateProfileSpec(logr.Discard(), profile)
		if err != nil {
			t.Fatalf("Unexpected validation error: %v", err)
		}
		tlsConfigFn, _ := tlspkg.NewTLSConfigFromProfile(validated)
		tlsOpts := []func(*tls.Config){tlsConfigFn, tlspkg.SetNextProtos(nextProtosForProfile(validated)...)}
		cfg := &tls.Config{}
		for _, opt := range tlsOpts {
			opt(cfg)
		}
		return cfg
	}

	cases := map[string]struct {
		profile     configv1.TLSProfileSpec
		wantProtos  []string
		wantCiphers []uint16
	}{
		"TLS 1.2 AES-256-only": {
			profile: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS12,
				Ciphers:       []string{"ECDHE-RSA-AES256-GCM-SHA384"},
			},
			wantProtos:  []string{"http/1.1"},
			wantCiphers: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
		},
		"TLS 1.2 AES-128-GCM": {
			profile: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS12,
				Ciphers:       []string{"ECDHE-RSA-AES128-GCM-SHA256"},
			},
			wantProtos:  tlspkg.HTTP2NextProtos,
			wantCiphers: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		"empty ciphers": {
			profile:    configv1.TLSProfileSpec{MinTLSVersion: configv1.VersionTLS12},
			wantProtos: tlspkg.HTTP2NextProtos,
		},
		"TLS 1.3": {
			profile: configv1.TLSProfileSpec{
				MinTLSVersion: configv1.VersionTLS13,
				Ciphers:       []string{"ECDHE-RSA-AES256-GCM-SHA384"},
			},
			wantProtos: tlspkg.HTTP2NextProtos,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := apply(t, tc.profile)
			if !slices.Equal(cfg.NextProtos, tc.wantProtos) {
				t.Errorf("NextProtos = %v, want %v", cfg.NextProtos, tc.wantProtos)
			}
			if tc.wantCiphers != nil && !slices.Equal(cfg.CipherSuites, tc.wantCiphers) {
				t.Errorf("CipherSuites = %#x, want %#x", cfg.CipherSuites, tc.wantCiphers)
			}
		})
	}

	t.Run("old profile honors TLS 1.0 minimum and prunes unusable ciphers", func(t *testing.T) {
		cfg := apply(t, *configv1.TLSProfiles[configv1.TLSProfileOldType])
		if cfg.MinVersion != tls.VersionTLS10 {
			t.Errorf("MinVersion = %#x, want %#x (configured minimum must be honored exactly)", cfg.MinVersion, tls.VersionTLS10)
		}
		if len(cfg.CipherSuites) == 0 {
			t.Fatal("Expected a non-empty pruned cipher suite list")
		}
		for _, id := range cfg.CipherSuites {
			if id >= 0x1300 && id <= 0x13ff {
				t.Errorf("CipherSuites contains TLS 1.3-only suite %#x", id)
			}
		}
	})
}
