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

package main

import (
	"errors"
	"flag"
	"os"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	admissionv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	platformcache "github.com/opendatahub-io/odh-platform-utilities/pkg/cache"
	platformlabels "github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
	"github.com/opendatahub-io/trainer-operator/internal/controller"
	// +kubebuilder:scaffold:imports
)

const trainerPartOf = "trainer"

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))

	utilruntime.Must(componentsv1alpha1.AddToScheme(scheme))
	utilruntime.Must(trainerv1alpha1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

func main() {
	var metricsAddr string
	var probeAddr string
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metrics endpoint binds to. "+
		"Use :8080 for HTTP or leave as 0 to disable the metrics service.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// Scope the cache to the applications namespace to avoid watching
	// resources cluster-wide. Injected by the ODH platform operator.
	// When unset (standalone/dev), nil falls back to all namespaces.
	var cacheNamespaces map[string]cache.Config
	if appNS := os.Getenv("APPLICATIONS_NAMESPACE"); appNS != "" {
		cacheNamespaces = map[string]cache.Config{appNS: {}}
		setupLog.Info("scoping cache to applications namespace", "namespace", appNS)
	}

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress:   metricsAddr,
			SecureServing: false,
		},
		HealthProbeBindAddress: probeAddr,
		// Cache hardening: without label selectors, informer caches store
		// every object of each watched type — even those unrelated to this
		// operator. A user with edit permissions can create enough large
		// objects to OOMKill the operator. Label selectors ensure only
		// trainer-managed resources are cached.
		// See: https://developers.redhat.com/articles/2026/06/01/protect-your-kubernetes-operator-oomkill
		Cache: cache.Options{
			DefaultTransform:  platformcache.StripUnusedFields(),
			DefaultNamespaces: cacheNamespaces,
			ByObject: map[client.Object]cache.ByObject{
				&appsv1.Deployment{}: {
					Label: labels.SelectorFromSet(labels.Set{platformlabels.PlatformPartOf: trainerPartOf}),
				},
				&corev1.Service{}: {
					Label: labels.SelectorFromSet(labels.Set{platformlabels.PlatformPartOf: trainerPartOf}),
				},
				&corev1.ConfigMap{}: {
					Label: labels.SelectorFromSet(labels.Set{platformlabels.PlatformPartOf: trainerPartOf}),
				},
				&admissionv1.ValidatingWebhookConfiguration{}: {
					Label: labels.SelectorFromSet(labels.Set{platformlabels.PlatformPartOf: trainerPartOf}),
				},
			},
		},
		// Disable cache for Namespace to prevent an implicit cluster-wide
		// informer from the ensureNamespace() Get call (anti-pattern 3).
		Client: client.Options{
			Cache: &client.CacheOptions{
				DisableFor: []client.Object{
					&corev1.Namespace{},
				},
			},
		},
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	const manifestsPath = "/opt/manifests-template"
	if fi, err := os.Stat(manifestsPath); err != nil {
		setupLog.Error(err, "manifests path is not accessible", "path", manifestsPath)
		os.Exit(1)
	} else if !fi.IsDir() {
		setupLog.Error(errors.New("not a directory"), "manifests path is not a directory", "path", manifestsPath)
		os.Exit(1)
	}

	const runtimesPath = "/opt/runtimes-template"
	if fi, err := os.Stat(runtimesPath); err != nil {
		setupLog.Error(err, "runtimes path is not accessible", "path", runtimesPath)
		os.Exit(1)
	} else if !fi.IsDir() {
		setupLog.Error(errors.New("not a directory"), "runtimes path is not a directory", "path", runtimesPath)
		os.Exit(1)
	}

	const imageStreamsPath = "/opt/imagestreams-template"
	if fi, err := os.Stat(imageStreamsPath); err != nil {
		setupLog.Error(err, "imagestreams path is not accessible", "path", imageStreamsPath)
		os.Exit(1)
	} else if !fi.IsDir() {
		setupLog.Error(errors.New("not a directory"), "imagestreams path is not a directory", "path", imageStreamsPath)
		os.Exit(1)
	}

	const workDir = "/opt/manifests-work"
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		setupLog.Error(err, "unable to create work directory", "path", workDir)
		os.Exit(1)
	}

	ctx := ctrl.SetupSignalHandler()

	if _, err := controller.NewReconciler(ctx, mgr, &controller.ReconcilerConfig{
		ManifestsPath:    manifestsPath,
		RuntimesPath:     runtimesPath,
		ImageStreamsPath: imageStreamsPath,
		WorkDir:          workDir,
	}); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Trainer")
		os.Exit(1)
	}
	// +kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
