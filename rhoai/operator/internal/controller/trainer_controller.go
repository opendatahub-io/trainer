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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/controller/conditions"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/controller/gc"
	"github.com/opendatahub-io/odh-platform-utilities/pkg/metadata/labels"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
)

const (
	trainerPartOf         = "trainer"
	defaultNamespace      = "opendatahub"
	finalizerName         = "components.platform.opendatahub.io/trainer-cleanup"
	readyCondition        = string(common.ConditionTypeReady)
	provisioningCondition = string(common.ConditionTypeProvisioningSucceeded)
)

type TrainerReconciler struct {
	client.Client
	Scheme          *runtime.Scheme
	ManifestsPath   string
	DynamicClient   dynamic.Interface
	DiscoveryClient discovery.DiscoveryInterface
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

func (r *TrainerReconciler) SetupWithManager(mgr ctrl.Manager) error {
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

	if err := r.ensureNamespace(ctx, namespace); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("NamespaceFailed"), conditions.WithError(err))
		return r.updateStatus(ctx, trainer, common.PhaseNotReady)
	}

	if err := r.renderAndApply(ctx); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("ProvisioningFailed"), conditions.WithError(err))
		return r.updateStatus(ctx, trainer, common.PhaseNotReady)
	}

	cm.MarkTrue(provisioningCondition, conditions.WithReason("Provisioned"), conditions.WithMessage("Trainer resources provisioned successfully"))

	return r.updateStatus(ctx, trainer, common.PhaseReady)
}

func (r *TrainerReconciler) reconcileRemoved(ctx context.Context, trainer *componentsv1alpha1.Trainer) (ctrl.Result, error) {
	log := logf.FromContext(ctx)
	log.Info("Trainer is marked as Removed, cleaning up managed resources")

	cm := newConditionManager(trainer)

	if err := r.runGC(ctx, trainer); err != nil {
		cm.MarkFalse(provisioningCondition, conditions.WithReason("CleanupFailed"), conditions.WithError(err))
		return r.updateStatus(ctx, trainer, common.PhaseNotReady)
	}

	cm.MarkFalse(provisioningCondition, conditions.WithReason("Removed"), conditions.WithMessage("Trainer has been removed"))

	return r.updateStatus(ctx, trainer, common.PhaseNotReady)
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

// --- Helpers ---

func (r *TrainerReconciler) renderAndApply(ctx context.Context) error {
	log := logf.FromContext(ctx)

	workDir := r.ManifestsPath + "-work"

	if err := ensureWorkDir(r.ManifestsPath, workDir); err != nil {
		return fmt.Errorf("preparing work directory: %w", err)
	}

	resources, err := renderManifests(workDir)
	if err != nil {
		return fmt.Errorf("rendering manifests: %w", err)
	}

	log.Info("Applying Trainer resources", "count", len(resources))

	if err := applyResources(ctx, r.Client, resources); err != nil {
		return fmt.Errorf("applying resources: %w", err)
	}

	return nil
}

func (r *TrainerReconciler) runGC(ctx context.Context, trainer *componentsv1alpha1.Trainer) error {
	namespace := resolveNamespace(trainer)

	collector := gc.New(
		gc.WithOnlyCollectOwned(false),
		gc.InNamespace(namespace),
	)

	return collector.Run(ctx, gc.RunParams{
		Client:          r.Client,
		DynamicClient:   r.DynamicClient,
		DiscoveryClient: r.DiscoveryClient,
		Owner:           trainer,
	})
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

func (r *TrainerReconciler) updateStatus(ctx context.Context, trainer *componentsv1alpha1.Trainer, phase common.Phase) (ctrl.Result, error) {
	trainer.Status.Phase = phase
	trainer.Status.ObservedGeneration = trainer.Generation

	if err := r.Status().Update(ctx, trainer); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update Trainer status: %w", err)
	}

	return ctrl.Result{}, nil
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
