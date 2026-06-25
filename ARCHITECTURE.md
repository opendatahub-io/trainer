# Architecture

This document describes the architecture of the Kubeflow Trainer (V2) operator, the midstream fork maintained by the OpenDataHub / RHOAI team. The upstream source lives at [kubeflow/trainer](https://github.com/kubeflow/trainer); this repository (`opendatahub-io/trainer`) adds RHOAI-specific extensions.

## High-level overview

The operator manages three CRDs and produces [JobSet](https://jobset.sigs.k8s.io/) workloads. The key idea is a **separation of concerns**: data scientists define _what_ to train via a `TrainJob`, while platform engineers define _how_ to train via `TrainingRuntime` / `ClusterTrainingRuntime` templates.

```
TrainJob (user-facing)
   │
   │ runtimeRef
   ▼
TrainingRuntime / ClusterTrainingRuntime (platform-defined template)
   │
   │ controller + plugins
   ▼
JobSet → Jobs → Pods (actual workload)
```

## CRDs

All three CRDs belong to the `trainer.kubeflow.org` API group, version `v1alpha1`.

### TrainJob

Namespaced resource. The data scientist's entry point. Key spec fields:

| Field | Purpose |
|---|---|
| `runtimeRef` | References a TrainingRuntime (namespaced) or ClusterTrainingRuntime (cluster-scoped) by name and kind |
| `trainer` | Overrides for the trainer container: image, command, args, env, numNodes, resourcesPerNode, numProcPerNode |
| `initializer` | Optional dataset and model init containers with storage URIs and secrets |
| `suspend` | Pause/resume the training job |
| `managedBy` | Controller ownership — `trainer.kubeflow.org/trainjob-controller` (default) or `kueue.x-k8s.io/multikueue` |
| `podTemplateOverrides` | Per-job overrides for pod spec (nodeSelector, tolerations, volumes, etc.) |

Status tracks conditions (`Suspended`, `Complete`, `Failed`) and per-job status counts.

Type definition: `pkg/apis/trainer/v1alpha1/trainjob_types.go`

### TrainingRuntime

Namespaced resource. Reusable training configuration template. Contains:

- `mlPolicy` — ML framework config: `torch` (PyTorch distributed/elastic) or `mpi` (Horovod/DeepSpeed). Mutually exclusive.
- `podGroupPolicy` — Gang scheduling via `coscheduling` (scheduler-plugins) or `volcano`.
- `template` — A `JobSetSpec` defining the actual jobs and pods.

Type definition: `pkg/apis/trainer/v1alpha1/trainingruntime_types.go`

### ClusterTrainingRuntime

Cluster-scoped version of TrainingRuntime. Shares the same `TrainingRuntimeSpec`. Can be referenced by TrainJobs in any namespace.

## Controllers

All three controllers live in `pkg/controller/` and are wired together in `pkg/controller/setup.go`.

### TrainJobReconciler

The main reconciler. On each reconciliation:

1. Looks up the referenced runtime via the runtime registry
2. Calls `runtime.NewObjects()` to generate Kubernetes ApplyConfigurations (JobSet + any framework-specific resources)
3. Applies each object with server-side apply (`FieldOwner("trainer")`)
4. Reconciles NetworkPolicy for pod network isolation (`pkg/rhai/networkpolicy.go`)
5. Updates TrainJob status by delegating to `runtime.TrainJobStatus()`
6. Runs RHOAI progression tracking if enabled via annotation
7. Notifies runtime reconcilers of every create/update/delete event (watcher pattern)

### TrainingRuntimeReconciler / ClusterTrainingRuntimeReconciler

These reconcilers protect runtimes from deletion while in use. They:

1. List all TrainJobs referencing the runtime
2. Add a `resource-in-use` finalizer when TrainJobs reference the runtime
3. Remove the finalizer when no TrainJobs reference it anymore

They receive TrainJob lifecycle events from the TrainJobReconciler via the watcher pattern, triggering re-evaluation with a 1-second delay.

## Runtime framework

The runtime framework is a **plugin-based architecture** in `pkg/runtime/`.

### Runtime interface

```go
type Runtime interface {
    NewObjects(ctx, trainJob)           -> []ApplyConfiguration
    RuntimeInfo(trainJob, template, …)  -> *Info
    TrainJobStatus(ctx, trainJob)       -> *TrainJobStatus
    EventHandlerRegistrars()            -> []ReconcilerBuilder
    ValidateObjects(ctx, old, new)      -> (Warnings, ErrorList)
}
```

### Runtime registry

`pkg/runtime/core/registry.go` maps `apiGroup/kind` to runtime factories:

| Key | Factory |
|---|---|
| `trainer.kubeflow.org/TrainingRuntime` | `NewTrainingRuntime` |
| `trainer.kubeflow.org/ClusterTrainingRuntime` | `NewClusterTrainingRuntime` (wraps TrainingRuntime) |

### Info struct

`pkg/runtime/runtime.go` defines `Info` — an intermediate representation that plugins read and mutate. Contains extracted labels, annotations, ML/PodGroup policies, a JobSet apply-configuration template, and an abstract `PodSet` list (name, ancestor role, replica count, containers, resource requirements).

### Plugin interfaces

Defined in `pkg/runtime/framework/interface.go`:

| Interface | Purpose |
|---|---|
| `EnforceMLPolicyPlugin` | Inject framework-specific env vars and config (e.g., TorchRun `PET_*` vars) |
| `EnforcePodGroupPolicyPlugin` | Configure gang scheduling (coscheduling or volcano PodGroups) |
| `CustomValidationPlugin` | Validate framework-specific constraints in webhooks |
| `PodNetworkPlugin` | Identify pod network topology |
| `ComponentBuilderPlugin` | Generate additional K8s resources (e.g., MPI SSH secrets, hostfile ConfigMaps) |
| `TrainJobStatusPlugin` | Compute job status from child resources |
| `WatchExtensionPlugin` | Register additional controller watches |

### Built-in plugins

Registered in `pkg/runtime/framework/plugins/registry.go`:

| Plugin | Interfaces | Purpose |
|---|---|---|
| **Torch** | EnforceMLPolicy, CustomValidation | PyTorch distributed training — injects `PET_*` env vars, validates numProcPerNode, supports TorchTune |
| **MPI** | EnforceMLPolicy, CustomValidation, ComponentBuilder, WatchExtension | MPI training — generates SSH key Secrets and hostfile ConfigMaps |
| **PlainML** | EnforceMLPolicy | Fallback for non-distributed training |
| **Coscheduling** | EnforcePodGroupPolicy | scheduler-plugins PodGroup creation |
| **Volcano** | EnforcePodGroupPolicy | Volcano PodGroup creation |
| **JobSet** | ComponentBuilder, TrainJobStatus | Builds the core JobSet resource and computes status from child Jobs |

### Reconciliation flow

```
TrainJob created/updated
  │
  ▼
TrainJobReconciler.Reconcile()
  │
  ├─ Look up Runtime from registry
  │
  ├─ runtime.NewObjects():
  │    ├─ Fetch TrainingRuntime/ClusterTrainingRuntime from cluster
  │    ├─ RuntimeInfo() → build Info struct
  │    ├─ Run EnforceMLPolicy plugins → mutate Info
  │    ├─ Run EnforcePodGroupPolicy plugins → mutate Info
  │    ├─ Run PodNetwork plugins → mutate Info
  │    ├─ Run ComponentBuilder plugins → generate ApplyConfigurations
  │    └─ Return list of objects to apply
  │
  ├─ Server-side apply each object
  ├─ Reconcile NetworkPolicy
  ├─ Update status via runtime.TrainJobStatus()
  └─ Run RHOAI progression tracking
```

## Webhooks

Validating webhooks in `pkg/webhooks/`:

- **TrainJob**: Validates that the referenced runtime exists; delegates to `runtime.ValidateObjects()` which runs all `CustomValidationPlugin`s
- **TrainingRuntime / ClusterTrainingRuntime**: Validates JobSet template structure — jobs with ancestor labels must have `replicas: 1`, and must contain the required container name (`node`, `dataset-initializer`, or `model-initializer`)

## Python initializers

`pkg/initializers/` contains init-container implementations for downloading datasets and models before training starts.

### Provider architecture

```
pkg/initializers/
├── types/types.py          # Dataclass configs (one per provider × resource type)
├── dataset/
│   ├── __main__.py         # URI scheme → provider dispatch
│   ├── huggingface.py      # hf:// scheme
│   ├── s3.py               # s3:// scheme
│   └── cache.py            # cache:// scheme (Ray-based distributed caching)
├── model/
│   ├── __main__.py         # URI scheme → provider dispatch
│   ├── huggingface.py      # hf:// scheme
│   └── s3.py               # s3:// scheme
└── utils/
    ├── utils.py            # Abstract base classes (DatasetProvider, ModelProvider)
    └── opendal.py          # S3 abstraction via Apache OpenDAL
```

All providers implement `DatasetProvider` or `ModelProvider` (load config from env vars, download to `/workspace/dataset` or `/workspace/model`). URI scheme dispatch uses Python 3.10+ `match/case`.

### Python API models

`api/python_api/` contains auto-generated Pydantic V2 models for all Kubernetes types in the CRDs. Generated from `api/openapi-spec/swagger.json` via OpenAPI Generator. Do not edit manually — run `make generate` to regenerate.

## Rust data cache

`pkg/data_cache/` contains a distributed data caching service written in Rust:

- **Head service** (`src/head/`) — coordinates workers, manages metadata, uses Apache Iceberg table format
- **Worker service** (`src/worker/`) — stores and serves data partitions, uses Apache Arrow columnar format and DataFusion query engine
- **Transport** — Arrow Flight (gRPC-based) for high-performance data transfer between nodes

Built as two binaries (`head` and `worker`) from `pkg/data_cache/cmd/`. The `cache://` initializer scheme in the Python initializers creates a LeaderWorkerSet that runs these services.

## RHOAI extensions

`pkg/rhai/` contains midstream-specific features not present in upstream Kubeflow:

- **Progression tracking** (`pkg/rhai/progression/`) — polls training metrics from pods and stores progress in TrainJob annotations. Enabled per-job via `trainer.opendatahub.io/progression-tracking: enabled`
- **NetworkPolicy** (`pkg/rhai/networkpolicy.go`) — creates NetworkPolicy resources for pod network isolation during training
- **RHOAI manifests** (`manifests/rhoai/`) — kustomize overlays with RHOAI-specific images, RBAC, and runtimes

## Directory layout

```
cmd/
├── trainer-controller-manager/   # Main Go binary — controller manager entrypoint
├── data_cache/                   # Dockerfile for Rust data cache service
├── initializers/                 # Dockerfile + requirements for Python init containers
├── runtimes/                     # Container configs for DeepSpeed, MLX runtimes
└── trainers/                     # Container configs for TorchTune trainers

pkg/
├── apis/
│   ├── trainer/v1alpha1/         # CRD type definitions + generated code
│   └── config/v1alpha1/          # Controller configuration types
├── controller/                   # Kubernetes controllers (TrainJob, TrainingRuntime, ClusterTrainingRuntime)
├── webhooks/                     # Validating webhooks
├── runtime/
│   ├── core/                     # Runtime registry and TrainingRuntime/ClusterTrainingRuntime implementations
│   ├── framework/
│   │   ├── core/                 # Plugin framework orchestration
│   │   └── plugins/              # Built-in plugins (torch, mpi, plainml, jobset, coscheduling, volcano)
│   └── indexer/                  # Field indexers for efficient TrainJob → Runtime lookups
├── initializers/                 # Python init containers (dataset/model providers)
├── data_cache/                   # Rust data cache service
├── rhai/                         # RHOAI-specific extensions (progression, network policy)
├── apply/                        # Server-side apply helpers
├── client/                       # Generated Go client (informers, listers)
├── config/                       # Controller configuration loading
├── constants/                    # Shared constants (labels, env vars, paths)
└── util/                         # Utilities (cert management, testing helpers)

manifests/
├── base/                         # Base kustomize manifests (CRDs, RBAC, webhook, runtimes)
├── rhoai/                        # RHOAI-specific kustomize overlays
└── third-party/                  # External dependencies

charts/kubeflow-trainer/          # Helm chart (depends on jobset subchart)
docs/proposals/                   # Design proposals (KEP-style)
```