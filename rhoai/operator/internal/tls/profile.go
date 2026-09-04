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
	"fmt"
	"time"

	"github.com/go-logr/logr"
	configv1 "github.com/openshift/api/config/v1"
	tlspkg "github.com/openshift/controller-runtime-common/pkg/tls"
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
// options for controller-runtime servers.
//
// NoMatch / NotFound fall back to Intermediate (non-OpenShift or CRD absent).
// Transient API errors use Intermediate and still register the watcher so a
// later profile is not dropped. Unexpected errors (Forbidden, etc.) fail closed.
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

	tlsConfigFn, unsupported := tlspkg.NewTLSConfigFromProfile(profile)
	if len(unsupported) > 0 {
		logger.Info("TLS profile contains unsupported ciphers", "unsupported", unsupported)
	}
	result.TLSOpts = append(result.TLSOpts, tlsConfigFn, tlspkg.SetNextProtos(tlspkg.HTTP2NextProtos...))

	if err := fetchAdherence(bootstrapCtx, k8sClient, result, logger); err != nil {
		return nil, err
	}

	return result, nil
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

// SetupWatcher registers a SecurityProfileWatcher that cancels the manager
// context when the cluster TLS profile or adherence policy changes.
// No-ops when ProfileFetched is false (non-OpenShift).
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
