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

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/blang/semver/v4"
	"gopkg.in/yaml.v3"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/cluster"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/controller/gc"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"

	fwapi "github.com/opendatahub-io/odh-platform-utilities/framework/api"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/actions"
	fwdeploy "github.com/opendatahub-io/odh-platform-utilities/framework/controller/actions/deploy"
	fwgc "github.com/opendatahub-io/odh-platform-utilities/framework/controller/actions/gc"
	fwdeployments "github.com/opendatahub-io/odh-platform-utilities/framework/controller/actions/status/deployments"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/conditions"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/handlers"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/predicates/dependent"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/reconciler"
	"github.com/opendatahub-io/odh-platform-utilities/framework/controller/types"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
)

const (
	TrainerInstanceName = "default-trainer"

	trainerPartOf    = "trainer"
	defaultNamespace = "opendatahub"
	finalizerName    = "components.platform.opendatahub.io/trainer-cleanup"

	degradedCondition = string(common.ConditionTypeDegraded)

	trainerKubeflowGroup   = "trainer.kubeflow.org"
	trainerKubeflowVersion = "v1alpha1"
	clusterTrainingRuntime = "ClusterTrainingRuntime"

	platformConfigMapName = "odh-trainer-config"
	platformVersionKey    = "platformVersion"
	platformReleaseName   = "platform"
)

var (
	jobSetOperatorGVK = schema.GroupVersionKind{
		Group:   jobSetOperatorGroup,
		Version: jobSetOperatorAPIVersion,
		Kind:    jobSetOperatorKind,
	}

	clusterTrainingRuntimeGVK = schema.GroupVersionKind{
		Group:   trainerKubeflowGroup,
		Version: trainerKubeflowVersion,
		Kind:    "ClusterTrainingRuntime",
	}
	trainingRuntimeGVK = schema.GroupVersionKind{
		Group:   trainerKubeflowGroup,
		Version: trainerKubeflowVersion,
		Kind:    "TrainingRuntime",
	}
)

type componentRelease struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
	RepoURL string `yaml:"repoUrl"`
}

type componentMetadata struct {
	Releases []componentRelease `yaml:"releases"`
}

// ReconcilerConfig holds paths and configuration for the trainer reconciler.
type ReconcilerConfig struct {
	ManifestsPath    string
	RuntimesPath     string
	ImageStreamsPath string
	WorkDir          string
}

type trainerActions struct {
	apiReader        client.Reader
	manifestsPath    string
	runtimesPath     string
	imageStreamsPath string
	workDir          string
	reconciler       *reconciler.Reconciler
}

// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch;update;watch
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups="",resources=limitranges,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list
// +kubebuilder:rbac:groups="",resources=secrets,verbs=create;get;list;patch;update;watch
// +kubebuilder:rbac:groups="",resources=services;serviceaccounts;configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=create;get;list;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings;rolebindings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=monitoring.coreos.com,resources=podmonitors,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=admissionregistration.k8s.io,resources=validatingwebhookconfigurations,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apiextensions.k8s.io,resources=customresourcedefinitions,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=jobset.x-k8s.io,resources=jobsets,verbs=create;get;list;patch;update;watch
// +kubebuilder:rbac:groups=networking.k8s.io,resources=networkpolicies,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=node.k8s.io,resources=runtimeclasses,verbs=get;list;watch
// +kubebuilder:rbac:groups=scheduling.volcano.sh,resources=podgroups,verbs=create;get;list;patch;update;watch
// +kubebuilder:rbac:groups=scheduling.x-k8s.io,resources=podgroups,verbs=create;get;list;patch;update;watch
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=clustertrainingruntimes;trainingruntimes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=clustertrainingruntimes/status;trainingruntimes/status,verbs=get
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=clustertrainingruntimes/finalizers;trainingruntimes/finalizers;trainjobs/finalizers,verbs=get;patch;update
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=trainjobs,verbs=create;delete;get;list;patch;update;watch
// +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=trainjobs/status,verbs=get;patch;update
// +kubebuilder:rbac:groups=image.openshift.io,resources=imagestreams,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=config.openshift.io,resources=apiservers;clusterversions,verbs=get
// +kubebuilder:rbac:groups=operators.coreos.com,resources=operatorconditions,verbs=list
// +kubebuilder:rbac:groups=operator.openshift.io,resources=jobsetoperators,verbs=get;list;watch

// NewReconciler creates a framework-based reconciler for the Trainer CR.
func NewReconciler(ctx context.Context, mgr ctrl.Manager, cfg *ReconcilerConfig) (*reconciler.Reconciler, error) {
	m := &trainerActions{
		apiReader:        mgr.GetAPIReader(),
		manifestsPath:    cfg.ManifestsPath,
		runtimesPath:     cfg.RuntimesPath,
		imageStreamsPath: cfg.ImageStreamsPath,
		workDir:          cfg.WorkDir,
	}

	nsFn := func(_ context.Context, rr *types.ReconciliationRequest) (string, error) {
		return resolveNamespace(rr.Instance.(*componentsv1alpha1.Trainer)), nil
	}

	r, err := reconciler.ReconcilerFor(mgr, &componentsv1alpha1.Trainer{}).
		Watches(&appsv1.Deployment{}).
		Watches(&corev1.Service{}).
		Watches(&corev1.ConfigMap{}).
		Watches(&admissionv1.ValidatingWebhookConfiguration{}).
		WatchesGVK(jobSetOperatorGVK,
			reconciler.WithEventHandler(handlers.ToNamed(TrainerInstanceName)),
			reconciler.WithPredicates(dependent.New(dependent.WithWatchStatus(true))),
			reconciler.Dynamic(reconciler.CrdExists(jobSetOperatorGVK))).
		WithAction(m.checkDependencies).
		WithAction(m.ensureNamespace).
		WithAction(m.updateReleases).
		WithAction(m.renderManifests).
		WithAction(withImmutableFieldRecovery(fwdeploy.NewAction(
			fwdeploy.WithCache(),
			fwdeploy.WithFieldOwner("trainer-module-controller"),
			fwdeploy.WithApplyOrder(),
		))).
		WithAction(fwdeployments.NewAction(fwdeployments.InNamespaceFn(nsFn))).
		// GC deletes stale resources from previous versions that are no longer
		// rendered. OnlyCollectOwned is false because cluster-scoped resources
		// (e.g. ClusterTrainingRuntimes) can't have owner references to the
		// namespace-scoped Trainer CR. Training runtimes are marked unremovable
		// here — they're cleaned up explicitly in the finalizer via the dynamic
		// client to handle the upstream trainer controller's resource-in-use
		// finalizer.
		WithAction(fwgc.NewAction(nsFn,
			fwgc.WithOnlyCollectOwned(false),
			fwgc.WithUnremovables(clusterTrainingRuntimeGVK, trainingRuntimeGVK))).
		WithFinalizer(m.cleanup).
		WithConditions("DeploymentsAvailable").
		WithReconcilerOpts(reconciler.WithFinalizerName(finalizerName)).
		Build(ctx)
	if err != nil {
		return nil, err
	}

	rel, readErr := readBootstrapRelease(cfg.ManifestsPath)
	if readErr != nil {
		logf.FromContext(ctx).V(1).Info("no bootstrap release; deferring to platform ConfigMap", "error", readErr)
	} else {
		r.Release = rel
	}

	m.reconciler = r

	return r, nil
}

// --- Action functions ---

func (m *trainerActions) checkDependencies(ctx context.Context, rr *types.ReconciliationRequest) error {
	log := logf.FromContext(ctx)
	c := rr.Client

	clusterType, err := cluster.DetectClusterType(ctx, c)
	if err != nil {
		rr.Conditions.MarkTrue(degradedCondition,
			conditions.WithReason("ClusterDetectionFailed"),
			conditions.WithMessage("Failed to detect cluster type: %v", err))
		return fmt.Errorf("failed to detect cluster type: %w", err)
	}

	log.V(1).Info("Detected cluster type", "clusterType", clusterType)

	if clusterType == cluster.ClusterTypeOpenShift {
		if !checkJobSetOperatorInstalled(ctx, c) {
			return m.degradeDependency(rr, "JobSetOperatorNotInstalled", getJobSetOperatorNotInstalledMessage())
		}
		if !checkJobSetOperatorCR(ctx, c) {
			return m.degradeDependency(rr, "JobSetOperatorCRNotFound", getJobSetOperatorCRMissingMessage())
		}
		if _, err := checkJobSetOperatorHealth(ctx, c); err != nil {
			return m.degradeDependency(rr, "JobSetOperatorDegraded", err.Error())
		}
		if !checkJobSetAvailable(ctx, c) {
			return m.degradeDependency(rr, "JobSetCRDNotFound", getJobSetMissingMessageOpenShift())
		}
	} else {
		if !checkJobSetAvailable(ctx, c) {
			return m.degradeDependency(rr, "JobSetCRDNotFound", getJobSetMissingMessage())
		}
	}

	rr.Conditions.MarkFalse(degradedCondition,
		conditions.WithReason("DependenciesAvailable"),
		conditions.WithMessage("All required dependencies are available"))

	return nil
}

func (m *trainerActions) degradeDependency(rr *types.ReconciliationRequest, reason, message string) error {
	rr.Conditions.MarkTrue(degradedCondition,
		conditions.WithReason(reason),
		conditions.WithMessage("%s", message))
	return fmt.Errorf("dependency not met: %s", message)
}

func (m *trainerActions) ensureNamespace(ctx context.Context, rr *types.ReconciliationRequest) error {
	c := rr.Client
	namespace := resolveNamespace(rr.Instance.(*componentsv1alpha1.Trainer))

	ns := &corev1.Namespace{}
	if err := c.Get(ctx, client.ObjectKey{Name: namespace}, ns); err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to check namespace: %w", err)
		}

		ns = &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: namespace,
				Labels: map[string]string{
					labels.PlatformPartOf: trainerPartOf,
				},
			},
		}
		if err := c.Create(ctx, ns); err != nil {
			return fmt.Errorf("failed to create namespace: %w", err)
		}
		logf.FromContext(ctx).Info("Created namespace", "namespace", namespace)
	}

	return nil
}

func (m *trainerActions) renderManifests(ctx context.Context, rr *types.ReconciliationRequest) error {
	log := logf.FromContext(ctx)
	trainer := rr.Instance.(*componentsv1alpha1.Trainer)
	namespace := resolveNamespace(trainer)

	clusterType, err := cluster.DetectClusterType(ctx, rr.Client)
	if err != nil {
		return fmt.Errorf("failed to detect cluster type: %w", err)
	}

	resourceSets := []resourceSet{
		{
			name:         "manifests",
			subDir:       "manifests",
			templatePath: m.manifestsPath,
			paramMap:     trainerImageParamMap,
			namespace:    namespace,
			overlay:      defaultOverlay,
		},
		{
			name:             "runtimes",
			subDir:           "runtimes",
			templatePath:     m.runtimesPath,
			paramMap:         runtimesParamMap,
			filterConfigMaps: true,
		},
	}

	if clusterType == cluster.ClusterTypeOpenShift {
		resourceSets = append(resourceSets, resourceSet{
			name:             "imagestreams",
			subDir:           "imagestreams",
			templatePath:     m.imageStreamsPath,
			paramMap:         imageStreamParamMap,
			namespace:        namespace,
			filterConfigMaps: true,
		})
	}

	for _, rs := range resourceSets {
		rendered, err := m.renderResourceSet(rs)
		if err != nil {
			return err
		}
		log.Info("Rendered "+rs.name, "count", len(rendered))
		rr.Resources = append(rr.Resources, rendered...)
	}

	rr.Generated = true
	return nil
}

func (m *trainerActions) renderResourceSet(rs resourceSet) ([]unstructured.Unstructured, error) {
	workDir := filepath.Join(m.workDir, rs.subDir)

	if err := ensureWorkDir(rs.templatePath, workDir); err != nil {
		return nil, fmt.Errorf("preparing %s work directory: %w", rs.name, err)
	}

	renderPath := workDir
	if rs.overlay != "" {
		renderPath = filepath.Join(workDir, rs.overlay)
	}

	if err := applyParamOverrides(renderPath, rs.paramMap); err != nil {
		return nil, fmt.Errorf("resolving %s params: %w", rs.name, err)
	}

	rendered, err := renderOverlay(renderPath, rs.namespace)
	if err != nil {
		return nil, fmt.Errorf("rendering %s: %w", rs.name, err)
	}

	if rs.filterConfigMaps {
		rendered = filterConfigMaps(rendered)
	}

	return rendered, nil
}

func (m *trainerActions) updateReleases(ctx context.Context, rr *types.ReconciliationRequest) error {
	log := logf.FromContext(ctx)
	trainer := rr.Instance.(*componentsv1alpha1.Trainer)
	namespace := resolveNamespace(trainer)

	releases, err := m.getComponentReleases()
	if err != nil {
		log.V(1).Info("Failed to read component releases", "error", err)
	} else {
		for _, release := range releases {
			trainer.Status.Releases = appendOrUpdateRelease(trainer.Status.Releases, release)
		}
	}

	platformVersion, err := m.getPlatformVersion(ctx, namespace)
	if err != nil {
		return fmt.Errorf("failed to get platform version: %w", err)
	}

	if platformVersion != "" {
		currentVersion := currentPlatformVersion(trainer)
		if currentVersion != platformVersion {
			// Perform any version-specific upgrade logic here (e.g.
			// schema migrations, operand pre-rollout checks). While
			// upgrade work is in progress, return an error so the
			// version is not advanced in status.
			log.Info("Platform version changed", "from", currentVersion, "to", platformVersion)
		}
		v, parseErr := semver.ParseTolerant(platformVersion)
		if parseErr != nil {
			log.V(1).Info("Ignoring unparsable platform version", "version", platformVersion, "error", parseErr)
		} else {
			trainer.Status.Releases = appendOrUpdateRelease(trainer.Status.Releases, common.ComponentRelease{
				Name:    platformReleaseName,
				Version: platformVersion,
			})
			m.reconciler.Release = fwapi.Release{
				Name:    fwapi.Platform(platformReleaseName),
				Version: v,
			}
		}
	}

	return nil
}

func (m *trainerActions) cleanup(ctx context.Context, rr *types.ReconciliationRequest) error {
	log := logf.FromContext(ctx)
	trainer := rr.Instance.(*componentsv1alpha1.Trainer)
	namespace := resolveNamespace(trainer)

	log.Info("Cleaning up trainer resources")

	cleanupTrainerResources(ctx, rr.Controller.GetDynamicClient())

	collector := gc.New(
		gc.WithOnlyCollectOwned(false),
		gc.InNamespace(namespace),
		gc.WithUnremovables(clusterTrainingRuntimeGVK, trainingRuntimeGVK),
	)

	return collector.Run(ctx, gc.RunParams{
		Client:          rr.Client,
		DynamicClient:   rr.Controller.GetDynamicClient(),
		DiscoveryClient: rr.Controller.GetDiscoveryClient(),
		Owner:           trainer,
	})
}

// withImmutableFieldRecovery wraps a deploy action to handle immutable field
// errors. When a Deployment has an immutable field change (e.g. selector),
// the API server rejects the SSA apply. This wrapper catches the error,
// deletes the offending Deployment, and returns an error to trigger a
// requeue — the next reconcile recreates it with the new spec.
func withImmutableFieldRecovery(deployAction actions.Fn) actions.Fn {
	return func(ctx context.Context, rr *types.ReconciliationRequest) error {
		err := deployAction(ctx, rr)
		if err == nil || !isImmutableFieldError(err) {
			return err
		}

		log := logf.FromContext(ctx)

		// The deploy action error doesn't identify which Deployment failed,
		// so we delete all Deployments. In practice there is only one
		// (kubeflow-trainer-controller-manager).
		for i := range rr.Resources {
			res := &rr.Resources[i]
			if res.GetKind() != "Deployment" {
				continue
			}

			toDelete := &unstructured.Unstructured{}
			toDelete.SetGroupVersionKind(res.GroupVersionKind())
			toDelete.SetName(res.GetName())
			toDelete.SetNamespace(res.GetNamespace())

			if delErr := rr.Client.Delete(ctx, toDelete); delErr != nil && !errors.IsNotFound(delErr) {
				log.Error(delErr, "Failed to delete Deployment for immutable field recovery",
					"name", res.GetName(), "namespace", res.GetNamespace())
				continue
			}

			log.Info("Deleted Deployment due to immutable field change, will recreate on next reconcile",
				"name", res.GetName(), "namespace", res.GetNamespace())
		}

		return err
	}
}

func isImmutableFieldError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "field is immutable")
}

// --- Helpers ---

type resourceSet struct {
	name             string
	subDir           string
	templatePath     string
	paramMap         map[string]string
	namespace        string
	overlay          string
	filterConfigMaps bool
}

func cleanupTrainerResources(ctx context.Context, dynClient dynamic.Interface) {
	log := logf.FromContext(ctx)

	gvrs := []schema.GroupVersionResource{
		{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "clustertrainingruntimes"},
		{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "trainingruntimes"},
	}

	selector := labels.PlatformPartOf + "=" + trainerPartOf
	propagation := metav1.DeletePropagationBackground

	for _, gvr := range gvrs {
		items, err := dynClient.Resource(gvr).Namespace("").List(ctx, metav1.ListOptions{
			LabelSelector: selector,
		})
		if err != nil {
			log.V(1).Info("Cannot list trainer resources for cleanup, skipping", "gvr", gvr, "error", err)
			continue
		}

		for i := range items.Items {
			item := &items.Items[i]
			if !item.GetDeletionTimestamp().IsZero() {
				continue
			}

			log.Info("Deleting trainer resource", "gvr", gvr, "name", item.GetName(), "namespace", item.GetNamespace())

			err := dynClient.Resource(gvr).Namespace(item.GetNamespace()).Delete(ctx, item.GetName(), metav1.DeleteOptions{
				PropagationPolicy: &propagation,
			})
			if err != nil && !errors.IsNotFound(err) {
				log.V(1).Info("Failed to delete trainer resource, skipping", "gvr", gvr, "name", item.GetName(), "error", err)
			}
		}
	}
}

func (m *trainerActions) getPlatformVersion(ctx context.Context, namespace string) (string, error) {
	cm := &corev1.ConfigMap{}
	if err := m.apiReader.Get(ctx, client.ObjectKey{Name: platformConfigMapName, Namespace: namespace}, cm); err != nil {
		if errors.IsNotFound(err) {
			return "", nil
		}
		return "", fmt.Errorf("failed to get platform config ConfigMap: %w", err)
	}

	return cm.Data[platformVersionKey], nil
}

// readBootstrapRelease reads component_metadata.yaml to provide a non-empty
// Release for the framework's deploy/GC annotation tracking at startup.
// This is a bootstrap value — the authoritative Release is set by
// updateReleases from the platform ConfigMap once it becomes available.
func readBootstrapRelease(manifestsPath string) (fwapi.Release, error) {
	metadataPath := filepath.Join(manifestsPath, "component_metadata.yaml")

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return fwapi.Release{}, fmt.Errorf("reading component metadata: %w", err)
	}

	var metadata componentMetadata
	if err := yaml.Unmarshal(data, &metadata); err != nil {
		return fwapi.Release{}, fmt.Errorf("parsing component metadata: %w", err)
	}

	if len(metadata.Releases) == 0 {
		return fwapi.Release{}, fmt.Errorf("no releases found in component metadata")
	}

	v, err := semver.ParseTolerant(metadata.Releases[0].Version)
	if err != nil {
		return fwapi.Release{}, fmt.Errorf("parsing release version %q: %w", metadata.Releases[0].Version, err)
	}

	return fwapi.Release{
		Name:    fwapi.Platform(metadata.Releases[0].Name),
		Version: v,
	}, nil
}

func (m *trainerActions) getComponentReleases() ([]common.ComponentRelease, error) {
	metadataPath := filepath.Join(m.manifestsPath, "component_metadata.yaml")

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("reading component metadata: %w", err)
	}

	var metadata componentMetadata
	if err := yaml.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("parsing component metadata: %w", err)
	}

	releases := make([]common.ComponentRelease, len(metadata.Releases))
	for i, r := range metadata.Releases {
		releases[i] = common.ComponentRelease{
			Name:    r.Name,
			Version: r.Version,
			RepoURL: r.RepoURL,
		}
	}

	return releases, nil
}

func currentPlatformVersion(trainer *componentsv1alpha1.Trainer) string {
	for _, r := range trainer.Status.Releases {
		if r.Name == platformReleaseName {
			return r.Version
		}
	}
	return ""
}

func appendOrUpdateRelease(releases []common.ComponentRelease, release common.ComponentRelease) []common.ComponentRelease {
	for i, r := range releases {
		if r.Name == release.Name {
			releases[i] = release
			return releases
		}
	}
	return append(releases, release)
}

func resolveNamespace(trainer *componentsv1alpha1.Trainer) string {
	if trainer.Spec.AppNamespace != "" {
		return trainer.Spec.AppNamespace
	}
	return defaultNamespace
}
