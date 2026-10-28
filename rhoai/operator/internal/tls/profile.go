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
	"time"

	"github.com/go-logr/logr"
	frameworktls "github.com/opendatahub-io/odh-platform-utilities/framework/tls"
	configv1 "github.com/openshift/api/config/v1"
	commonTLS "github.com/openshift/controller-runtime-common/pkg/tls"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const bootstrapTimeout = 10 * time.Second

type Result struct {
	Profile        configv1.TLSProfileSpec
	ProfileHonored bool
	Adherence      configv1.TLSAdherencePolicy
	UsedFallback   bool
	Watchable      bool
	TLSOpts        []func(*tls.Config)
}

func Resolve(ctx context.Context, k8sClient client.Client, logger logr.Logger) (*Result, error) {
	bootstrapCtx, cancel := context.WithTimeout(ctx, bootstrapTimeout)
	defer cancel()

	loaded, err := frameworktls.LoadWithAdherence(bootstrapCtx, k8sClient)
	if err != nil {
		return nil, fmt.Errorf("loading APIServer TLS profile and adherence: %w", err)
	}

	profileHonored := frameworktls.ShouldHonorClusterTLSProfile(loaded.AdherencePolicy)
	effectiveProfile := loaded.Spec
	if !profileHonored {
		effectiveProfile = *configv1.TLSProfiles[configv1.TLSProfileIntermediateType]
	}

	tlsConfigFn, unsupported, err := frameworktls.ConfigFromProfileWithError(effectiveProfile)
	if err != nil {
		return nil, fmt.Errorf("building TLS config from profile: %w", err)
	}
	if len(unsupported) > 0 {
		logger.Info("TLS profile contains unsupported ciphers or groups", "unsupported", unsupported)
	}

	return &Result{
		Profile:        loaded.Spec,
		ProfileHonored: profileHonored,
		Adherence:      loaded.AdherencePolicy,
		UsedFallback:   loaded.UsedFallback,
		Watchable:      loaded.Watchable,
		TLSOpts: []func(*tls.Config){
			tlsConfigFn,
			commonTLS.SetNextProtos(commonTLS.HTTP2NextProtos...),
		},
	}, nil
}

func SetupWatcher(mgr manager.Manager, result *Result, cancel context.CancelFunc, logger logr.Logger) error {
	if result == nil || !result.Watchable {
		return nil
	}

	watcher := newSecurityProfileWatcher(mgr.GetClient(), result, cancel, logger)
	if err := watcher.SetupWithManager(mgr); err != nil {
		return fmt.Errorf("setting up TLS profile watcher: %w", err)
	}
	return nil
}

func newSecurityProfileWatcher(k8sClient client.Client, result *Result, cancel context.CancelFunc, logger logr.Logger) *frameworktls.SecurityProfileWatcher {
	watcher := &frameworktls.SecurityProfileWatcher{
		Client:                    k8sClient,
		InitialTLSProfileSpec:     result.Profile,
		InitialTLSAdherencePolicy: result.Adherence,
		OnAdherencePolicyChange: func(_ context.Context, _, _ configv1.TLSAdherencePolicy) {
			logger.Info("TLS adherence policy changed, initiating shutdown to reload")
			cancel()
		},
	}
	if result.ProfileHonored || result.UsedFallback {
		watcher.OnProfileChange = func(_ context.Context, _, _ configv1.TLSProfileSpec) {
			logger.Info("TLS profile changed, initiating shutdown to reload")
			cancel()
		}
	}
	return watcher
}
