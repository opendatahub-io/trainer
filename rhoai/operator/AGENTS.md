# odh-trainer-operator

Trainer v2 Module Controller for ODH modular architecture (RHAISTRAT-1064). Reconciles the `Trainer` Module CR (`components.platform.opendatahub.io/v1alpha1`) and deploys upstream Kubeflow Trainer resources via kustomize rendering from `opendatahub-io/trainer` manifests.

## Structure

- `api/v1alpha1/` - CRD types implementing `common.PlatformObject` from `odh-platform-utilities`
- `internal/controller/` - Reconciler, manifest rendering, params.env handling
- `cmd/main.go` - Operator entrypoint
- `config/` - Kustomize manifests (CRDs, RBAC, manager deployment, samples)
- `hack/` - Manifest collection script (`get_trainer_manifests.sh`)
- `manifests/runtimes/` - ClusterTrainingRuntime definitions (torch, training-hub)
- `manifests/imagestreams/` - ImageStream definitions (CUDA, ROCm, CPU training images)
- `test/e2e/` - End-to-end tests (Kind cluster)
- `test/support/` - Shared test client (`Client` wrapping `kubernetes.Interface`)
- `test/utils/` - Shell command utilities (make, kind, cert-manager)

## Key Paths

- `api/v1alpha1/trainer_types.go` - Trainer CRD with PlatformObject interface, ManagementSpec, CEL singleton validation
- `internal/controller/trainer_controller.go` - Reconciler: managed/removed/delete paths, finalizer, GC, conditions
- `internal/controller/manifests.go` - Kustomize rendering pipeline, SSA apply via `resources.Apply`
- `internal/controller/params.go` - [TEMPORARY] params.env parsing for RELATED_IMAGE override
- `hack/get_trainer_manifests.sh` - Fetches upstream trainer manifests (ODH or RHOAI)
- `config/samples/components_v1alpha1_trainer.yaml` - Singleton CR `default-trainer`

## Architecture

### Platform Utilities (`odh-platform-utilities`)

The controller uses shared utilities from `opendatahub-io/odh-platform-utilities`:
- `api/common` - PlatformObject interface, ManagementSpec, Status, Condition types
- `pkg/controller/conditions` - Condition Manager with happiness recomputation
- `pkg/controller/gc` - GC Collector for label-based resource cleanup via discovery API
- `pkg/render/kustomize` - Kustomize manifest rendering
- `pkg/resources` - Server-Side Apply
- `pkg/metadata/labels` - `platform.opendatahub.io/part-of` label

### Manifest Pipeline

1. Build time: `get_trainer_manifests.sh` fetches upstream manifests into `opt/manifests/`
2. Dockerfile: copies upstream manifests into `/opt/manifests-template/`, runtimes into `/opt/runtimes-template/`, imagestreams into `/opt/imagestreams-template/`
3. Runtime: copies templates to writable work dir, applies RELATED_IMAGE env var overrides to params.env, renders kustomize overlay, applies with SSA

### Reconcile Flow

- **Managed**: ensure finalizer → ensure namespace → render manifests → SSA apply → update status (Ready)
- **Removed**: ensure finalizer → GC cleanup (label-based discovery) → update status (NotReady)
- **Deleted**: GC cleanup → remove finalizer → CR deleted

## Development

### Prerequisites

- Go 1.25+
- Podman (or Docker via `CONTAINER_TOOL=docker`)
- kubectl v1.28+

### Common Commands

```bash
make manifests        # Generate CRDs and RBAC from markers
make generate         # Generate DeepCopy methods
make fmt              # Format code
make vet              # Vet code
make lint             # Run linter
make test             # Run unit tests (envtest)
make build            # Build the operator binary
make run              # Run operator locally against cluster
make docker-build     # Build container image (uses podman by default)
make install          # Install CRDs into cluster
make deploy IMG=<img> # Deploy operator to cluster
```

### Running Tests

Unit tests (controller tests with envtest):
```bash
make test
```

E2E tests (creates a Kind cluster, deploys the operator, runs tests, tears down):
```bash
make test-e2e
```

## Writing Code

### CRD Changes

1. Edit `api/v1alpha1/trainer_types.go`
2. Run `make manifests generate` to regenerate CRDs and DeepCopy methods
3. Update controller logic in `internal/controller/trainer_controller.go`

### RBAC

RBAC rules are derived from `// +kubebuilder:rbac:` markers in the controller. After adding new markers, run `make manifests` to regenerate.

The controller SA must hold all permissions that upstream trainer ClusterRoles grant — Kubernetes RBAC escalation prevention blocks creating a ClusterRole with permissions the creator doesn't already have. When upstream manifests add new ClusterRole rules, matching RBAC markers must be added to the controller.

### License Headers

All `.go` files must include the Apache 2.0 license header from `hack/boilerplate.go.txt`. Do not remove existing headers.

### Before Committing

Run `make lint` after any code changes.

### Tests

Tests use standard Go testing with gomega matchers (no ginkgo). Use `TestMain` for setup/teardown, `t.Run` for subtests, `t.Cleanup` for resource cleanup, and `gomega.NewWithT(t)` for assertions. Test names use camelCase.

Controller unit tests use envtest (lightweight API server). Test files live alongside the controller in `internal/controller/`.

E2E tests in `test/e2e/` deploy the operator to a Kind cluster and verify the full lifecycle including metrics. Tests use a Go client (`test/support.Client`) for cluster interactions instead of shelling out to kubectl. Test files are split by concern: `controller_test.go`, `metrics_test.go`.
