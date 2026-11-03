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
	"reflect"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
)

const testECDHERSAAES128GCM = "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"

func TestTranslateProfileIntermediate(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])
	if err != nil {
		t.Fatal(err)
	}

	if opts.MinVersion != "1.2" {
		t.Errorf("MinVersion = %q, want 1.2", opts.MinVersion)
	}
	want := []string{
		"TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
		testECDHERSAAES128GCM,
		"TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
		"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
		"TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
		"TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
	}
	if !reflect.DeepEqual(opts.CipherSuites, want) {
		t.Errorf("CipherSuites = %v, want %v", opts.CipherSuites, want)
	}
}

func TestTranslateProfileModernOmitsCipherSuites(t *testing.T) {
	// The Modern profile's cipher list is TLS 1.3 suites only; Go
	// negotiates those regardless of CipherSuites, so the translated
	// list is empty and the field must be omitted to keep the operand's
	// startup validation green.
	opts, err := TranslateProfile(t.Context(), *configv1.TLSProfiles[configv1.TLSProfileModernType])
	if err != nil {
		t.Fatal(err)
	}

	if opts.MinVersion != "1.3" {
		t.Errorf("MinVersion = %q, want 1.3", opts.MinVersion)
	}
	if len(opts.CipherSuites) != 0 {
		t.Errorf("CipherSuites = %v, want empty", opts.CipherSuites)
	}
}

func TestTranslateProfileOldKeepsLegacySuites(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), *configv1.TLSProfiles[configv1.TLSProfileOldType])
	if err != nil {
		t.Fatal(err)
	}

	if opts.MinVersion != "1.0" {
		t.Errorf("MinVersion = %q, want 1.0", opts.MinVersion)
	}
	for _, name := range opts.CipherSuites {
		if name == "TLS_RSA_WITH_3DES_EDE_CBC_SHA" {
			return
		}
	}
	t.Errorf("CipherSuites = %v, want TLS_RSA_WITH_3DES_EDE_CBC_SHA retained from the Old profile", opts.CipherSuites)
}

func TestTranslateProfileFiltersRC4(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Ciphers: []string{
			"TLS_ECDHE_RSA_WITH_RC4_128_SHA",
			"DES-CBC3-SHA",
			"ECDHE-RSA-AES128-GCM-SHA256",
		},
	})

	want := []string{"TLS_RSA_WITH_3DES_EDE_CBC_SHA", "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"}
	if !reflect.DeepEqual(opts.CipherSuites, want) {
		t.Errorf("CipherSuites = %v, want %v (RC4 dropped, order preserved)", opts.CipherSuites, want)
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestTranslateProfileCustomTLS13NamesDropped(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Ciphers: []string{
			"TLS_AES_128_GCM_SHA256",
			"TLS_CHACHA20_POLY1305_SHA256",
			"ECDHE-RSA-AES256-GCM-SHA384",
		},
	})

	want := []string{"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"}
	if !reflect.DeepEqual(opts.CipherSuites, want) {
		t.Errorf("CipherSuites = %v, want %v (TLS 1.3 names dropped)", opts.CipherSuites, want)
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestTranslateProfileGoNamesPassThrough(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Ciphers:       []string{testECDHERSAAES128GCM},
	})

	want := []string{testECDHERSAAES128GCM}
	if !reflect.DeepEqual(opts.CipherSuites, want) {
		t.Errorf("CipherSuites = %v, want %v (Go names pass through)", opts.CipherSuites, want)
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestTranslateProfileEmptyVersionOmitted(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), configv1.TLSProfileSpec{})
	if opts.MinVersion != "" {
		t.Errorf("MinVersion = %q, want empty", opts.MinVersion)
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(opts.CipherSuites) != 0 {
		t.Errorf("CipherSuites = %v, want empty", opts.CipherSuites)
	}
}

func TestTranslateProfileProjectsCurvePreferences(t *testing.T) {
	profile := configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Ciphers:       []string{"ECDHE-RSA-AES128-GCM-SHA256"},
		Groups:        []configv1.TLSGroup{configv1.TLSGroupX25519MLKEM768, configv1.TLSGroupX25519},
	}
	opts, err := TranslateProfile(t.Context(), profile)
	if err != nil {
		t.Fatal(err)
	}
	// X25519MLKEM768 = 4588, X25519 = 29 (Go crypto/tls CurveIDs).
	if !reflect.DeepEqual(opts.CurvePreferences, []int32{4588, 29}) {
		t.Errorf("CurvePreferences = %v, want [4588 29]", opts.CurvePreferences)
	}
}

func TestTranslateProfileOmitsCurvePreferencesWithoutGroups(t *testing.T) {
	opts, err := TranslateProfile(t.Context(), configv1.TLSProfileSpec{MinTLSVersion: configv1.VersionTLS12})
	if err != nil {
		t.Fatal(err)
	}
	if len(opts.CurvePreferences) != 0 {
		t.Errorf("CurvePreferences = %v, want empty when the profile has no groups", opts.CurvePreferences)
	}
}

func TestTranslateProfileIntermediateProjectsGroupIDs(t *testing.T) {
	// The Intermediate profile specifies X25519MLKEM768, X25519, P-256 and
	// P-384, in that order (Go CurveIDs 4588, 29, 23, 24).
	opts, err := TranslateProfile(t.Context(), *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(opts.CurvePreferences, []int32{4588, 29, 23, 24}) {
		t.Errorf("CurvePreferences = %v, want [4588 29 23 24]", opts.CurvePreferences)
	}
}

func TestTranslateProfileFailsClosedOnUnsupportedGroups(t *testing.T) {
	profile := configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Groups:        []configv1.TLSGroup{"NotAGroup"},
	}
	if _, err := TranslateProfile(t.Context(), profile); err == nil {
		t.Fatal("want error when the profile specifies groups but none are supported")
	}
}
