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
	"crypto/tls"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	configv1 "github.com/openshift/api/config/v1"
	tlspkg "github.com/openshift/controller-runtime-common/pkg/tls"
	libgocrypto "github.com/openshift/library-go/pkg/crypto"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const bootstrapTimeout = 10 * time.Second

// Result holds the resolved TLS configuration from the cluster profile.
type Result struct {
	Profile          configv1.TLSProfileSpec
	ProfileFetched   bool
	Adherence        configv1.TLSAdherencePolicy
	AdherenceFetched bool
	TLSOpts          []func(*tls.Config)
}

// Resolve fetches the cluster TLS profile and adherence policy, returning TLS
// options for controller-runtime servers. The shared TLS implementation owns
// profile, cipher, and group conversion.
func Resolve(ctx context.Context, k8sClient client.Client, logger logr.Logger) (*Result, error) {
	bootstrapCtx, cancel := context.WithTimeout(ctx, bootstrapTimeout)
	defer cancel()

	result := &Result{}
	profile, err := tlspkg.FetchAPIServerTLSProfile(bootstrapCtx, k8sClient)
	if err != nil {
		if fallbackErr := handleProfileFetchError(err, result, logger); fallbackErr != nil {
			return nil, fallbackErr
		}
		profile = *configv1.TLSProfiles[configv1.TLSProfileIntermediateType]
	} else {
		result.ProfileFetched = true
	}
	result.Profile = profile

	validated, err := validateProfileSpec(logger, profile)
	if err != nil {
		return nil, fmt.Errorf("validating TLS profile: %w", err)
	}
	tlsConfigFn, unsupported := tlspkg.NewTLSConfigFromProfile(validated)
	if len(unsupported) > 0 {
		logger.Info("TLS profile contains unsupported ciphers", "unsupported", unsupported)
	}
	result.TLSOpts = append(result.TLSOpts, tlsConfigFn, tlspkg.SetNextProtos(nextProtosForProfile(validated)...))

	if err := fetchAdherence(bootstrapCtx, k8sClient, result, logger); err != nil {
		return nil, err
	}
	return result, nil
}

// validateProfileSpec rejects TLS profiles that cannot be honored without a
// silent substitution: an unknown minimum version, or a cipher list with no
// suite usable for TLS <= 1.2 handshakes, is an error. Partially unusable
// cipher lists are pruned with a log line. The returned spec is safe to pass
// to tlspkg.NewTLSConfigFromProfile: it cannot panic on the minimum version
// and its ciphers cannot all resolve to nothing.
func validateProfileSpec(logger logr.Logger, profile configv1.TLSProfileSpec) (configv1.TLSProfileSpec, error) {
	minVersion, err := libgocrypto.TLSVersion(string(profile.MinTLSVersion))
	if err != nil {
		return profile, fmt.Errorf("invalid minimum TLS version %q: %w", profile.MinTLSVersion, err)
	}
	// Go 1.22+ defaults the server minimum to TLS 1.2 only when tls.Config.MinVersion
	// is unset; the explicit minimum set here needs no GODEBUG.
	if minVersion >= tls.VersionTLS13 || len(profile.Ciphers) == 0 {
		return profile, nil
	}

	usable := make([]string, 0, len(profile.Ciphers))
	var dropped []string
	for _, cipher := range profile.Ciphers {
		if cipherUsableForTLS12(cipher) {
			usable = append(usable, cipher)
		} else {
			dropped = append(dropped, cipher)
		}
	}
	if len(usable) == 0 {
		return profile, fmt.Errorf("no cipher suite usable with minimum TLS version %q: requested ciphers %v", profile.MinTLSVersion, profile.Ciphers)
	}
	if len(dropped) > 0 {
		logger.Info("Dropping TLS profile ciphers not usable for TLS <= 1.2 handshakes", "dropped", dropped)
		profile.Ciphers = usable
	}
	return profile, nil
}

func nextProtosForProfile(profile configv1.TLSProfileSpec) []string {
	if profileSupportsHTTP2(profile) {
		return tlspkg.HTTP2NextProtos
	}
	return []string{"http/1.1"}
}

func profileSupportsHTTP2(profile configv1.TLSProfileSpec) bool {
	minVersion, err := libgocrypto.TLSVersion(string(profile.MinTLSVersion))
	if err != nil {
		return false
	}
	if minVersion >= tls.VersionTLS13 || len(profile.Ciphers) == 0 {
		return true
	}

	for _, cipher := range profile.Ciphers {
		id, ok := cipherSuiteID(cipher)
		if ok && (id == tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 || id == tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256) {
			return true
		}
	}
	return false
}

// cipherUsableForTLS12 reports whether name, in IANA or OpenSSL form, resolves
// to a cipher suite ID that a TLS <= 1.2 handshake can negotiate. TLS 1.3-only
// suites (IDs 0x1300-0x13ff) never qualify.
func cipherUsableForTLS12(name string) bool {
	id, ok := cipherSuiteID(name)
	return ok && (id < 0x1300 || id > 0x13ff)
}

func cipherSuiteID(name string) (uint16, bool) {
	id, err := libgocrypto.CipherSuite(name)
	if err != nil {
		iana := libgocrypto.OpenSSLToIANACipherSuites([]string{name})
		if len(iana) != 1 {
			return 0, false
		}
		if id, err = libgocrypto.CipherSuite(iana[0]); err != nil {
			return 0, false
		}
	}
	return id, id != 0
}

func handleProfileFetchError(err error, result *Result, logger logr.Logger) error {
	switch {
	case apimeta.IsNoMatchError(err):
		logger.Info("TLS profile not available (non-OpenShift cluster)")
	case apierrors.IsNotFound(err):
		logger.Info("APIServer resource not found, using Intermediate defaults")
	case apierrors.IsServiceUnavailable(err),
		apierrors.IsTimeout(err),
		apierrors.IsServerTimeout(err),
		apierrors.IsTooManyRequests(err),
		errors.Is(err, context.DeadlineExceeded):
		logger.Info("Transient API error, using Intermediate defaults", "error", err)
		result.ProfileFetched = true
	default:
		return fmt.Errorf("reading APIServer TLS profile: %w", err)
	}
	return nil
}

func fetchAdherence(ctx context.Context, k8sClient client.Client, result *Result, logger logr.Logger) error {
	adherence, err := tlspkg.FetchAPIServerTLSAdherencePolicy(ctx, k8sClient)
	if err != nil {
		switch {
		case apimeta.IsNoMatchError(err):
			logger.Info("TLS adherence API not available (non-OpenShift or pre-4.22 cluster)")
		case apierrors.IsNotFound(err):
			logger.Info("APIServer resource not found for adherence, skipping")
		case apierrors.IsServiceUnavailable(err),
			apierrors.IsTimeout(err),
			apierrors.IsServerTimeout(err),
			apierrors.IsTooManyRequests(err),
			apierrors.IsInternalError(err),
			errors.Is(err, context.DeadlineExceeded):
			logger.Info("Transient error fetching TLS adherence policy, watcher will retry", "error", err)
			result.AdherenceFetched = true
		default:
			return fmt.Errorf("reading APIServer TLS adherence policy: %w", err)
		}
		return nil
	}
	result.AdherenceFetched = true
	result.Adherence = adherence
	return nil
}

// SetupWatcher registers a TLS profile watcher that cancels the manager when
// the profile or adherence policy changes. It is a no-op for non-OpenShift
// clusters where the profile API is unavailable.
func SetupWatcher(mgr manager.Manager, result *Result, cancel context.CancelFunc, logger logr.Logger) error {
	if result == nil || !result.ProfileFetched {
		return nil
	}

	watcher := &tlspkg.SecurityProfileWatcher{
		Client:                mgr.GetClient(),
		InitialTLSProfileSpec: result.Profile,
		OnProfileChange: func(_ context.Context, _, _ configv1.TLSProfileSpec) {
			logger.Info("TLS profile changed, initiating shutdown to reload")
			cancel()
		},
	}
	if result.AdherenceFetched {
		watcher.InitialTLSAdherencePolicy = result.Adherence
		watcher.OnAdherencePolicyChange = func(_ context.Context, _, _ configv1.TLSAdherencePolicy) {
			logger.Info("TLS adherence policy changed, initiating shutdown to reload")
			cancel()
		}
	}

	if err := watcher.SetupWithManager(mgr); err != nil {
		return fmt.Errorf("setting up TLS profile watcher: %w", err)
	}
	return nil
}
