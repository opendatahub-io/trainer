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
	stderrors "errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/cluster"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/controller/conditions"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/controller/gc"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
)

const (
	trainerPartOf         = "trainer"
	defaultNamespace      = "opendatahub"
	finalizerName         = "components.platform.opendatahub.io/trainer-cleanup"
	readyCondition        = string(common.ConditionTypeReady)
	provisioningCondition = string(common.ConditionTypeProvisioningSucceeded)
	degradedCondition     = string(common.ConditionTypeDegraded)

	trainerKubeflowGroup   = "trainer.kubeflow.org"
	trainerKubeflowVersion = "v1alpha1"

	// dependencyCheckInterval is how often to recheck for missing dependencies
	dependencyCheckInterval = 60 * time.Second
)

type componentRelease struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
	RepoURL string `yaml:"repoUrl"`
}

type componentMetadata struct {
	Releases []componentRelease `yaml:"releases"`
}

type TrainerReconciler struct {
	client.Client
	Scheme           *runtime.Scheme
	ManifestsPath    string
	RuntimesPath     string
	ImageStreamsPath string
	WorkDir          string
	DynamicClient    dynamic.Interface
	DiscoveryClient  discovery.DiscoveryInterface
}

// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=components.platform.opendatahub.io,resources=trainers/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch;update;watch
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
// +kubebuilder:rbac:groups=config.openshift.io,resources=clusterversions,verbs=get
// +kubebuilder:rbac:groups=operators.coreos.com,resources=operatorconditions,verbs=list
// +kubebuilder:rbac:groups=operator.openshift.io,resources=jobsetoperators,verbs=get;list

func (r *TrainerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// TODO(RHOAIENG-62940): add Watches for downstream resources to detect drift
	return ctrl.NewControllerManagedBy(mgr).
		For(&componentsv1alpha1.Trainer{}).
		Named("trainer").
		Complete(r)
}

func (r *TrainerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	trainer := &componentsv1alpha1.Trainer{}
	if err := r.Get(ctx, req.NamespacedName, trainer); err != nil {
		if errors.IsNotFound(err) {
			log.Info("Trainer resource not found, skipping reconciliation")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("failed to get Trainer: %w", err)
	}

	if !trainer.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(ctx, trainer)
	}

	if err := r.ensureFinalizer(ctx, trainer); err != nil {
		return ctrl.Result{}, err
	}

	if trainer.GetManagementState() == common.Removed {
		return r.reconcileRemoved(ctx, trainer)
	}

	return r.reconcileManaged(ctx, trainer)
}

// --- Reconcile paths ---

func (r *TrainerReconciler) reconcileManaged(ctx context.Context, trainer *componentsv1alpha1.Trainer) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	cm := newConditionManager(trainer)
	namespace := resolveNamespace(trainer)
	log.Info("Reconciling Trainer", "namespace", namespace)

	clusterType, err := cluster.DetectClusterType(ctx, r.Client)
	if err != nil {
		log.Error(err, "Failed to detect cluster type")
		cm.MarkTrue(degradedCondition,
			conditions.WithReason("ClusterDetectionFailed"),
			conditions.WithMessage("Failed to detect cluster type: %v", err))
		return ctrl.Result{}, stderrors.Join(err, r.updateStatus(ctx, trainer, common.PhaseNotReady))
	}

	log.V(1).Info("Detected cluster type", "clusterType", clusterType)

	if clusterType == cluster.ClusterTypeOpenShift {
		if result, done := r.checkOpenShiftDependencies(ctx, trainer, cm); done {
			return result, nil
		}
	} else {
		if result, done := r.checkKubernetesDependencies(ctx, trainer, cm); done {
			return result, nil
		}
	}

	// Dependencies available - module is not degraded
	cm.MarkFalse(degradedCondition,
		conditions.WithReason("DependenciesAvailable"),
		conditions.WithMessage("All required dependencies are available"))

	if err := r.ensureNamespace(ctx, namespace); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("NamespaceFailed"), conditions.WithError(err))
		return ctrl.Result{}, stderrors.Join(err, r.updateStatus(ctx, trainer, common.PhaseNotReady))
	}

	if err := r.renderAndApply(ctx, namespace, clusterType); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("ProvisioningFailed"), conditions.WithError(err))
		return ctrl.Result{}, stderrors.Join(err, r.updateStatus(ctx, trainer, common.PhaseNotReady))
	}

	if err := r.checkDeploymentHealth(ctx, namespace); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("DeploymentNotReady"), conditions.WithError(err))
		return ctrl.Result{}, stderrors.Join(err, r.updateStatus(ctx, trainer, common.PhaseNotReady))
	}

	cm.MarkTrue(provisioningCondition, conditions.WithReason("Provisioned"), conditions.WithMessage("Trainer resources provisioned successfully"))

	return ctrl.Result{}, r.updateStatus(ctx, trainer, common.PhaseReady)
}

func (r *TrainerReconciler) reconcileRemoved(ctx context.Context, trainer *componentsv1alpha1.Trainer) (ctrl.Result, error) {
	log := logf.FromContext(ctx)
	log.Info("Trainer is marked as Removed, cleaning up managed resources")

	cm := newConditionManager(trainer)

	if err := r.runGC(ctx, trainer); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("CleanupFailed"), conditions.WithError(err))
		return ctrl.Result{}, stderrors.Join(err, r.updateStatus(ctx, trainer, common.PhaseNotReady))
	}

	cm.MarkFalse(provisioningCondition, conditions.WithReason("Removed"), conditions.WithMessage("Trainer has been removed"))

	return ctrl.Result{}, r.updateStatus(ctx, trainer, common.PhaseNotReady)
}

func (r *TrainerReconciler) reconcileDelete(ctx context.Context, trainer *componentsv1alpha1.Trainer) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	if !controllerutil.ContainsFinalizer(trainer, finalizerName) {
		return ctrl.Result{}, nil
	}

	log.Info("Trainer is being deleted, cleaning up managed resources")

	if err := r.runGC(ctx, trainer); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to clean up managed resources: %w", err)
	}

	patch := client.MergeFrom(trainer.DeepCopy())
	controllerutil.RemoveFinalizer(trainer, finalizerName)
	if err := r.Patch(ctx, trainer, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to remove finalizer: %w", err)
	}

	log.Info("Finalizer removed, cleanup complete")
	return ctrl.Result{}, nil
}

func (r *TrainerReconciler) checkOpenShiftDependencies(ctx context.Context, trainer *componentsv1alpha1.Trainer, cm *conditions.Manager) (ctrl.Result, bool) {
	if !r.checkJobSetOperatorInstalled(ctx) {
		return r.handleMissingDependency(ctx, trainer, cm,
			"JobSet Operator",
			getJobSetOperatorNotInstalledMessage(),
			"Waiting for JobSet Operator to be installed")
	}

	if !r.checkJobSetOperatorCR(ctx) {
		return r.handleMissingDependency(ctx, trainer, cm,
			"JobSetOperator CR",
			getJobSetOperatorCRMissingMessage(),
			"Waiting for JobSetOperator CR to be created")
	}

	if !r.checkJobSetAvailable(ctx) {
		return r.handleMissingDependency(ctx, trainer, cm,
			"JobSet CRD",
			getJobSetMissingMessageOpenShift(),
			"Waiting for JobSet CRD to be installed")
	}

	return ctrl.Result{}, false
}

func (r *TrainerReconciler) checkKubernetesDependencies(ctx context.Context, trainer *componentsv1alpha1.Trainer, cm *conditions.Manager) (ctrl.Result, bool) {
	if !r.checkJobSetAvailable(ctx) {
		return r.handleMissingDependency(ctx, trainer, cm,
			"JobSet CRD",
			getJobSetMissingMessage(),
			"Waiting for JobSet CRD to be installed")
	}

	return ctrl.Result{}, false
}

func (r *TrainerReconciler) checkDeploymentHealth(ctx context.Context, namespace string) error {
	log := logf.FromContext(ctx)

	deployments := &appsv1.DeploymentList{}
	err := r.List(
		ctx,
		deployments,
		client.InNamespace(namespace),
		client.MatchingLabels{labels.PlatformPartOf: trainerPartOf},
	)
	if err != nil {
		return fmt.Errorf("failed to list deployments: %w", err)
	}

	if len(deployments.Items) == 0 {
		log.V(1).Info("No deployments found yet", "label", fmt.Sprintf("%s=%s", labels.PlatformPartOf, trainerPartOf))
		return nil
	}

	ready := 0
	var notReadyDeployments []string
	for _, deployment := range deployments.Items {
		if deployment.Status.ReadyReplicas == deployment.Status.Replicas && deployment.Status.Replicas != 0 {
			ready++
		} else {
			log.Info("Deployment not ready", "name", deployment.Name, "namespace", deployment.Namespace,
				"readyReplicas", deployment.Status.ReadyReplicas, "replicas", deployment.Status.Replicas)
			notReadyDeployments = append(notReadyDeployments, deployment.Namespace+"/"+deployment.Name)
		}
	}

	if ready != len(deployments.Items) {
		return fmt.Errorf("%d/%d deployments ready: %v", ready, len(deployments.Items), notReadyDeployments)
	}

	log.V(1).Info("All deployments are ready", "count", len(deployments.Items))
	return nil
}

// --- Helpers ---

func (r *TrainerReconciler) handleMissingDependency(
	ctx context.Context,
	trainer *componentsv1alpha1.Trainer,
	cm *conditions.Manager,
	dependencyName string,
	errorMessage string,
	waitingMessage string,
) (ctrl.Result, bool) {
	log := logf.FromContext(ctx)
	log.Info(dependencyName+" not available, will recheck", "recheckAfter", dependencyCheckInterval)

	cm.MarkTrue(degradedCondition,
		conditions.WithReason("DependencyMissing"),
		conditions.WithMessage("%s", errorMessage))
	cm.MarkFalse(provisioningCondition,
		conditions.WithReason("DependencyMissing"),
		conditions.WithMessage("%s", waitingMessage))

	if err := r.updateStatus(ctx, trainer, common.PhaseNotReady); err != nil {
		log.Error(err, "Failed to update Trainer status")
	}
	return ctrl.Result{RequeueAfter: dependencyCheckInterval}, true
}

type resourceSet struct {
	name             string
	subDir           string
	templatePath     string
	paramMap         map[string]string
	namespace        string
	overlay          string
	filterConfigMaps bool
}

func (r *TrainerReconciler) renderAndApply(ctx context.Context, namespace string, clusterType cluster.ClusterType) error {
	if err := r.renderAndApplyResourceSet(ctx, resourceSet{
		name:         "manifests",
		subDir:       "manifests",
		templatePath: r.ManifestsPath,
		paramMap:     trainerImageParamMap,
		namespace:    namespace,
		overlay:      defaultOverlay,
	}); err != nil {
		return fmt.Errorf("applying Trainer manifests: %w", err)
	}

	if err := r.renderAndApplyResourceSet(ctx, resourceSet{
		name:             "runtimes",
		subDir:           "runtimes",
		templatePath:     r.RuntimesPath,
		paramMap:         runtimesParamMap,
		filterConfigMaps: true,
	}); err != nil {
		return fmt.Errorf("applying ClusterTrainingRuntimes: %w", err)
	}

	if clusterType == cluster.ClusterTypeOpenShift {
		if err := r.renderAndApplyResourceSet(ctx, resourceSet{
			name:             "imagestreams",
			subDir:           "imagestreams",
			templatePath:     r.ImageStreamsPath,
			paramMap:         imageStreamParamMap,
			namespace:        namespace,
			filterConfigMaps: true,
		}); err != nil {
			return fmt.Errorf("applying ImageStreams: %w", err)
		}
	}

	return nil
}

func (r *TrainerReconciler) renderAndApplyResourceSet(ctx context.Context, rs resourceSet) error {
	log := logf.FromContext(ctx)

	workDir := filepath.Join(r.WorkDir, rs.subDir)

	if err := ensureWorkDir(rs.templatePath, workDir); err != nil {
		return fmt.Errorf("preparing %s work directory: %w", rs.name, err)
	}

	renderPath := workDir
	if rs.overlay != "" {
		renderPath = filepath.Join(workDir, rs.overlay)
	}

	if err := applyParamOverrides(renderPath, rs.paramMap); err != nil {
		return fmt.Errorf("resolving %s params: %w", rs.name, err)
	}

	rendered, err := renderOverlay(renderPath, rs.namespace)
	if err != nil {
		return fmt.Errorf("rendering %s: %w", rs.name, err)
	}

	if rs.filterConfigMaps {
		rendered = filterConfigMaps(rendered)
	}

	log.Info("Applying "+rs.name, "count", len(rendered))

	if err := applyResources(ctx, r.Client, rendered); err != nil {
		return fmt.Errorf("applying %s: %w", rs.name, err)
	}

	return nil
}

func (r *TrainerReconciler) runGC(ctx context.Context, trainer *componentsv1alpha1.Trainer) error {
	namespace := resolveNamespace(trainer)

	r.cleanupTrainerResources(ctx)

	collector := gc.New(
		gc.WithOnlyCollectOwned(false),
		gc.InNamespace(namespace),
		gc.WithUnremovables(
			schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: "ClusterTrainingRuntime"},
			schema.GroupVersionKind{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Kind: "TrainingRuntime"},
		),
	)

	return collector.Run(ctx, gc.RunParams{
		Client:          r.Client,
		DynamicClient:   r.DynamicClient,
		DiscoveryClient: r.DiscoveryClient,
		Owner:           trainer,
	})
}

func (r *TrainerReconciler) cleanupTrainerResources(ctx context.Context) {
	log := logf.FromContext(ctx)

	gvrs := []schema.GroupVersionResource{
		{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "clustertrainingruntimes"},
		{Group: trainerKubeflowGroup, Version: trainerKubeflowVersion, Resource: "trainingruntimes"},
	}

	selector := labels.PlatformPartOf + "=" + trainerPartOf
	propagation := metav1.DeletePropagationBackground

	for _, gvr := range gvrs {
		items, err := r.DynamicClient.Resource(gvr).Namespace("").List(ctx, metav1.ListOptions{
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

			err := r.DynamicClient.Resource(gvr).Namespace(item.GetNamespace()).Delete(ctx, item.GetName(), metav1.DeleteOptions{
				PropagationPolicy: &propagation,
			})
			if err != nil && !errors.IsNotFound(err) {
				log.V(1).Info("Failed to delete trainer resource, skipping", "gvr", gvr, "name", item.GetName(), "error", err)
			}
		}
	}
}

func (r *TrainerReconciler) ensureFinalizer(ctx context.Context, trainer *componentsv1alpha1.Trainer) error {
	if controllerutil.ContainsFinalizer(trainer, finalizerName) {
		return nil
	}

	patch := client.MergeFrom(trainer.DeepCopy())
	controllerutil.AddFinalizer(trainer, finalizerName)
	if err := r.Patch(ctx, trainer, patch); err != nil {
		return fmt.Errorf("failed to add finalizer: %w", err)
	}

	logf.FromContext(ctx).Info("Added finalizer")
	return nil
}

func (r *TrainerReconciler) ensureNamespace(ctx context.Context, name string) error {
	ns := &corev1.Namespace{}
	if err := r.Get(ctx, client.ObjectKey{Name: name}, ns); err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to check namespace: %w", err)
		}

		ns = &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					labels.PlatformPartOf: trainerPartOf,
				},
			},
		}
		if err := r.Create(ctx, ns); err != nil {
			return fmt.Errorf("failed to create namespace: %w", err)
		}
		logf.FromContext(ctx).Info("Created namespace", "namespace", name)
	}

	return nil
}

func (r *TrainerReconciler) getComponentReleases() ([]common.ComponentRelease, error) {
	metadataPath := filepath.Join(r.ManifestsPath, "component_metadata.yaml")

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

func (r *TrainerReconciler) updateStatus(ctx context.Context, trainer *componentsv1alpha1.Trainer, phase common.Phase) error {
	log := logf.FromContext(ctx)

	trainer.Status.Phase = phase
	trainer.Status.ObservedGeneration = trainer.Generation

	releases, err := r.getComponentReleases()
	if err != nil {
		log.V(1).Info("Failed to read component releases", "error", err)
	} else {
		trainer.Status.Releases = releases
	}

	return r.Status().Update(ctx, trainer)
}

func newConditionManager(trainer *componentsv1alpha1.Trainer) *conditions.Manager {
	return conditions.NewManager(trainer, readyCondition, provisioningCondition)
}

func resolveNamespace(trainer *componentsv1alpha1.Trainer) string {
	if trainer.Spec.AppNamespace != "" {
		return trainer.Spec.AppNamespace
	}
	return defaultNamespace
}
