# odh-trainer-operator

Kubernetes operator for managing the Trainer component on OpenDataHub/RHOAI. Scaffolded with Kubebuilder (go.kubebuilder.io/v4), manages the `Trainer` custom resource in the `components.platform.opendatahub.io` API group. Built with Go, controller-runtime, Operator SDK.

## Structure

- `api/v1alpha1/` - CRD type definitions (`Trainer`, `TrainerSpec`, `TrainerStatus`)
- `internal/controller/` - Reconciler implementation (`TrainerReconciler`)
- `cmd/main.go` - Operator entrypoint
- `config/` - Kustomize manifests (CRDs, RBAC, manager deployment, samples)
- `test/e2e/` - End-to-end tests
- `test/support/` - Shared test client (`Client` wrapping `kubernetes.Interface`)
- `test/utils/` - Shell command utilities (make, kind, cert-manager)
- `hack/` - Development scripts

## Key Paths

- `api/v1alpha1/trainer_types.go` - Trainer CRD spec/status definitions
- `internal/controller/trainer_controller.go` - Main reconciliation logic
- `config/samples/` - Example Trainer CR manifests

## Development

### Prerequisites

- Go 1.24+
- Podman (or Docker via `CONTAINER_TOOL=docker`)
- kubectl v1.11.3+
- Access to a Kubernetes cluster

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

E2E tests (requires running cluster with operator deployed):
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

### Before Committing

Run `make lint` after any code changes to catch formatting and lint issues early.

### Tests

Tests use standard Go testing with gomega matchers (no ginkgo). Use `TestMain` for setup/teardown, `t.Run` for subtests, `t.Cleanup` for resource cleanup, and `gomega.NewWithT(t)` for assertions.

Controller unit tests use envtest (lightweight API server). Test files live alongside the controller in `internal/controller/`.

E2E tests in `test/e2e/` deploy the operator to a Kind cluster and verify the full lifecycle including metrics. Tests use a Go client (`test/support.Client`) for cluster interactions instead of shelling out to kubectl. Test files are split by concern: `controller_test.go`, `metrics_test.go`.
