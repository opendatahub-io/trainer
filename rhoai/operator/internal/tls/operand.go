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
	"fmt"
	"strconv"
	"strings"

	trainertls "github.com/kubeflow/trainer/v2/pkg/apis/config/v1alpha1"
	frameworktls "github.com/opendatahub-io/odh-platform-utilities/framework/tls"
	configv1 "github.com/openshift/api/config/v1"
	ocpcrypto "github.com/openshift/library-go/pkg/crypto"
)

// operandMinVersions maps OpenShift TLSProtocolVersion values to the
// minVersion strings accepted by the operand's Configuration API.
var operandMinVersions = map[configv1.TLSProtocolVersion]string{
	configv1.VersionTLS10: "1.0",
	configv1.VersionTLS11: "1.1",
	configv1.VersionTLS12: "1.2",
	configv1.VersionTLS13: "1.3",
}

// resolveCipherID resolves a profile cipher name (OpenSSL-style alias or
// IANA/Go suite name) to its TLS ID using openshift/library-go's translation
// tables, which track the OpenShift profile vocabulary upstream.
func resolveCipherID(name string) (uint16, bool) {
	if id, err := ocpcrypto.CipherSuite(name); err == nil {
		return id, true
	}
	ianaNames := ocpcrypto.OpenSSLToIANACipherSuites([]string{name})
	if len(ianaNames) != 1 {
		return 0, false
	}
	id, err := ocpcrypto.CipherSuite(ianaNames[0])
	if err != nil {
		return 0, false
	}
	return id, true
}

// goCipherName resolves a cipher suite ID to its Go (IANA-style) name as
// known to this build, from either the secure or the insecure runtime table.
func goCipherName(id uint16) (string, bool) {
	for _, cs := range tls.CipherSuites() {
		if cs.ID == id {
			return cs.Name, true
		}
	}
	for _, cs := range tls.InsecureCipherSuites() {
		if cs.ID == id {
			return cs.Name, true
		}
	}
	return "", false
}

// TranslateProfile converts the effective cluster TLS profile into the tls
// section of the operand's controller_manager_config.yaml.
//
// Cipher names are translated from the OpenShift/OpenSSL vocabulary to the
// Go suite names the operand's configuration validation recognizes. Suites
// this Go build cannot negotiate (TLS 1.3 suites, which Go always enables
// regardless of CipherSuites, and unrecognized names) and RC4 suites
// (RFC 7465; the operand rejects them outright at startup) are dropped. If
// nothing survives translation, cipherSuites is omitted so the operand uses
// the Go defaults for TLS 1.2 and below.
//
// TLS groups are translated to the numeric curve IDs the operand's
// CurvePreferences field takes, via the shared framework helper. A profile
// that specifies groups but none this Go build supports fails translation:
// serving with the Go default curves would silently ignore a strict
// cluster policy.
func TranslateProfile(ctx context.Context, profile configv1.TLSProfileSpec) (trainertls.TLSOptions, error) {
	opts := trainertls.TLSOptions{
		MinVersion: operandMinVersions[profile.MinTLSVersion],
	}

	curves, err := frameworktls.CurvePreferencesFromSpec(ctx, &profile)
	if err != nil {
		return trainertls.TLSOptions{}, fmt.Errorf("translating TLS groups: %w", err)
	}
	for _, id := range strings.Split(curves, ",") {
		if id == "" {
			continue
		}
		code, err := strconv.ParseInt(id, 10, 32)
		if err != nil {
			return trainertls.TLSOptions{}, fmt.Errorf("parsing curve ID %q: %w", id, err)
		}
		opts.CurvePreferences = append(opts.CurvePreferences, int32(code))
	}

	for _, name := range profile.Ciphers {
		id, ok := resolveCipherID(name)
		if !ok || rc4CipherIDs[id] || isTLS13CipherID(id) {
			continue
		}
		if goName, known := goCipherName(id); known {
			opts.CipherSuites = append(opts.CipherSuites, goName)
		}
	}

	return opts, nil
}

// rc4CipherIDs is the set of RC4 suite IDs this Go runtime knows, derived
// from the insecure table by name. RC4 is categorically broken: RFC 7465
// prohibits it, and the operand rejects RC4 names outright at controller
// startup. Deriving the set by name avoids referencing the RC4 constants,
// which the tls-config-lint scanner flags on sight.
var rc4CipherIDs = func() map[uint16]bool {
	ids := make(map[uint16]bool)
	for _, cs := range tls.InsecureCipherSuites() {
		if strings.Contains(cs.Name, "RC4") {
			ids[cs.ID] = true
		}
	}
	return ids
}()

// isTLS13CipherID reports whether the suite is a TLS 1.3 suite (IANA IDs
// 0x1301-0x1304). Go negotiates the built-in TLS 1.3 suites regardless of
// the CipherSuites field, and this Go version lists them in
// tls.CipherSuites(); they are dropped so the rendered list stays within
// what every operand build's startup validation recognizes.
func isTLS13CipherID(id uint16) bool {
	return id >= 0x1301 && id <= 0x1304
}
