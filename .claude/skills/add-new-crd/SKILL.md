# Adding a New CRD to Kubeflow Trainer

## When to use

Use this skill when adding a new Custom Resource Definition (CRD) to the Kubeflow Trainer project, following the same patterns as the existing TrainJob, TrainingRuntime, and ClusterTrainingRuntime CRDs.

## Steps

### 1. Define API types

Create or extend type definitions in `pkg/apis/trainer/v1alpha1/`:

- Add the new type in a `<resource>_types.go` file with proper kubebuilder markers:
  ```go
  // +genclient
  // +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
  // +kubebuilder:object:root=true
  // +kubebuilder:subresource:status          // if the resource has a status subresource
  // +kubebuilder:storageversion
  // +kubebuilder:resource:scope=Cluster      // only for cluster-scoped resources
  ```
- Add a corresponding `<Resource>List` type with `+kubebuilder:object:root=true`
- Define `<Resource>Spec` and `<Resource>Status` structs
- Use `// +kubebuilder:validation:*` markers for field-level validation
- Follow the immutability pattern: mark fields with `// IMMUTABLE` in comments and enforce in webhooks

Reference files:
- `pkg/apis/trainer/v1alpha1/trainjob_types.go` — namespaced CRD with status subresource
- `pkg/apis/trainer/v1alpha1/trainingruntime_types.go` — both namespaced and cluster-scoped CRDs sharing a spec type

### 2. Register the type with the scheme

Update `pkg/apis/trainer/v1alpha1/groupversion_info.go`:
- The existing `SchemeBuilder` auto-registers types annotated with `+kubebuilder:object:root=true`
- Also register in `cmd/trainer-controller-manager/main.go` if the type belongs to a new API group

### 3. Run code generation

```bash
make generate    # Generates deepcopy, defaults, OpenAPI, and client code
make manifests   # Generates CRD YAML, RBAC roles, and webhook manifests
```

Generated files:
- `pkg/apis/trainer/v1alpha1/zz_generated.deepcopy.go` — DeepCopyObject implementations
- `pkg/apis/trainer/v1alpha1/zz_generated.defaults.go` — defaulting functions
- `manifests/base/crds/trainer.kubeflow.org_<resource>.yaml` — CRD YAML
- `manifests/base/rbac/role.yaml` — updated ClusterRole
- `charts/kubeflow-trainer/crds/` — Helm chart CRD copy

### 4. Add a webhook

Create `pkg/webhooks/<resource>_webhook.go` following the existing pattern:

```go
type <Resource>Webhook struct {
    runtimes map[string]runtime.Runtime  // if runtime-aware validation is needed
}

// +kubebuilder:webhook:path=/validate-...,mutating=false,failurePolicy=fail,...
func (w *<Resource>Webhook) ValidateCreate(ctx, obj) (warnings, error)
func (w *<Resource>Webhook) ValidateUpdate(ctx, oldObj, newObj) (warnings, error)
func (w *<Resource>Webhook) ValidateDelete(ctx, obj) (warnings, error)
```

Register the webhook in `pkg/webhooks/setup.go` inside the `Setup()` function.

Reference: `pkg/webhooks/trainjob_webhook.go`, `pkg/webhooks/trainingruntime_webhook.go`

### 5. Add a controller

Create `pkg/controller/<resource>_controller.go`:

- Define a reconciler struct with `client.Client`, `APIReader`, and event recorder
- Implement `Reconcile(ctx, req) (ctrl.Result, error)`
- Add RBAC markers:
  ```go
  // +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=<resource>,verbs=get;list;watch;update;patch
  // +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=<resource>/status,verbs=get;update;patch
  // +kubebuilder:rbac:groups=trainer.kubeflow.org,resources=<resource>/finalizers,verbs=get;update;patch
  ```
- Implement `SetupWithManager(mgr)` with `ctrl.NewControllerManagedBy(mgr).For(&v1alpha1.<Resource>{}).Complete(r)`
- Register in `pkg/controller/setup.go` inside `SetupControllers()`

Reference: `pkg/controller/trainjob_controller.go` (main reconciler with runtime delegation)

### 6. Add runtime integration (if applicable)

If the new CRD participates in the runtime framework:

- Register in `pkg/runtime/core/registry.go` — add to `SupportedRuntimes()`
- Implement the `runtime.Runtime` interface in `pkg/runtime/core/`
- Add framework plugins if needed in `pkg/runtime/framework/plugins/`

### 7. Update Helm chart

- Copy the CRD YAML to `charts/kubeflow-trainer/crds/`
- Add any new RBAC rules to chart templates
- Update `charts/kubeflow-trainer/values.yaml` if new config is needed

### 8. Add tests

- **Unit tests**: Add to `pkg/controller/` and `pkg/webhooks/` for the new controller/webhook logic
- **Integration tests**: Add Ginkgo suites in `test/integration/` using envtest
- **E2E tests**: Add scenarios in `test/e2e/` if end-to-end validation is needed

Run tests:
```bash
make test               # Go unit tests
make test-integration   # Integration tests with envtest
make test-e2e           # End-to-end tests (requires cluster)
```

### 9. Update kustomize manifests

- Add CRD to `manifests/base/crds/kustomization.yaml`
- Update `manifests/base/rbac/` if RBAC was regenerated
- Add webhook configuration to `manifests/base/webhook/`

## Key patterns to follow

- Use `client.Apply()` with `FieldOwner("trainer")` for server-side apply (see TrainJobReconciler)
- Use the `resource-in-use` finalizer pattern to prevent deletion of resources that are referenced by other CRs
- Use the watcher pattern (see TrainingRuntimeReconciler) to react to changes in dependent resources
- Package-level markers go in `doc.go`: `+k8s:defaulter-gen=TypeMeta`, `+k8s:openapi-gen=true`, `+k8s:deepcopy-gen=package`