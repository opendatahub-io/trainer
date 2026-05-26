# trainer-operator

Standalone operator for Kubeflow Trainer v2. Reconciles the `Trainer` CR (`components.platform.opendatahub.io/v1alpha1`) and deploys upstream Kubeflow Trainer resources via kustomize rendering from [opendatahub-io/trainer](https://github.com/opendatahub-io/trainer) manifests.

## Architecture

### Manifest Pipeline

1. **Build time:** `hack/get_trainer_manifests.sh` fetches upstream manifests into `opt/manifests/`
2. **Dockerfile:** copies manifests into the container at `/opt/manifests-template/`
3. **Runtime:** copies to a writable work dir, applies `RELATED_IMAGE` env var overrides to `params.env`, renders `rhoai/` overlay via kustomize, applies with Server-Side Apply

### Reconcile Flow

- **Managed**: ensure finalizer → ensure namespace → render manifests → SSA apply → update status (Ready)
- **Removed**: ensure finalizer → GC cleanup (label-based discovery) → update status (NotReady)
- **Deleted**: GC cleanup → remove finalizer → CR deleted

### Platform Utilities

The controller uses shared utilities from [opendatahub-io/odh-platform-utilities](https://github.com/opendatahub-io/odh-platform-utilities):
- `api/common` — PlatformObject interface, ManagementSpec, Status, Condition types
- `pkg/controller/conditions` — Condition Manager with happiness recomputation
- `pkg/controller/gc` — GC Collector for label-based resource cleanup via discovery API
- `pkg/render/kustomize` — Kustomize manifest rendering
- `pkg/resources` — Server-Side Apply

## Structure

```
api/v1alpha1/        CRD types implementing common.PlatformObject
internal/controller/ Reconciler, manifest rendering, params.env handling
cmd/main.go          Operator entrypoint
config/              Kustomize manifests (CRDs, RBAC, manager deployment, samples)
hack/                Manifest collection script (get_trainer_manifests.sh)
manifests/           Training runtimes and ImageStream definitions
test/e2e/            End-to-end tests
test/support/        Shared test client
test/utils/          Shell command utilities
```

## Development

### Prerequisites

- Go 1.25+
- Podman (or Docker via `CONTAINER_TOOL=docker`)
- kubectl v1.28+
- Access to a Kubernetes v1.28+ cluster

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

### Deploy to Cluster

```bash
make docker-build docker-push IMG=<some-registry>/odh-trainer-operator:tag
make install
make deploy IMG=<some-registry>/odh-trainer-operator:tag
kubectl apply -k config/samples/
```

### Uninstall

```bash
kubectl delete -k config/samples/
make uninstall
make undeploy
```

## Contributing

### CRD Changes

1. Edit `api/v1alpha1/trainer_types.go`
2. Run `make manifests generate` to regenerate CRDs and DeepCopy methods
3. Update controller logic in `internal/controller/trainer_controller.go`

### RBAC

RBAC rules are derived from `// +kubebuilder:rbac:` markers in the controller. After adding new markers, run `make manifests` to regenerate.

The controller SA must hold all permissions that upstream trainer ClusterRoles grant — Kubernetes RBAC escalation prevention blocks creating a ClusterRole with permissions the creator doesn't already have.

### Before Committing

Run `make lint` after any code changes.

## License

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
