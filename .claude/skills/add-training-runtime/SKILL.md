# Adding a New TrainingRuntime or ClusterTrainingRuntime

## When to use

Use this skill when adding a new pre-built TrainingRuntime or ClusterTrainingRuntime manifest that defines a reusable training configuration (e.g., a new distributed framework, a new model fine-tuning recipe, or a new hardware-optimized setup).

## Background

TrainingRuntimes are templates that platform engineers create for data scientists. A TrainJob references a runtime via `runtimeRef`, and the controller merges the runtime's JobSet template with the TrainJob's overrides to produce the final workload.

- **TrainingRuntime** — namespaced, available only within one namespace
- **ClusterTrainingRuntime** — cluster-scoped, available to all namespaces (most common for pre-built runtimes)

## Steps

### 1. Understand the runtime spec structure

A runtime spec has three top-level sections:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime    # or TrainingRuntime
metadata:
  name: <runtime-name>
spec:
  mlPolicy:                     # ML framework config (optional)
    numNodes: 1
    torch:                      # OR mpi: (mutually exclusive)
      numProcPerNode: auto
  podGroupPolicy:               # Gang scheduling (optional)
    coscheduling:
      scheduleTimeoutSeconds: 60
  template:                     # JobSet template (required)
    spec:
      replicatedJobs:
        - name: <job-name>
          template:
            spec:
              template:
                spec:
                  containers:
                    - name: trainer
                      image: <image>
                      ...
```

### 2. Choose the right ML policy

**Torch (PyTorch distributed)**:
```yaml
mlPolicy:
  numNodes: 1
  torch:
    numProcPerNode: auto   # "auto", "cpu", "gpu", or integer
    # elasticPolicy:       # optional, for elastic training
    #   minNodes: 1
    #   maxNodes: 4
```

**MPI (Horovod, DeepSpeed with MPI)**:
```yaml
mlPolicy:
  numNodes: 1
  mpi:
    numProcPerNode: 1
    mpiImplementation: OpenMPI
    runLauncherAsNode: false
```

**No policy** — for plain ML workloads without distributed training.

### 3. Define the JobSet template

The `template.spec` follows the [JobSet API](https://jobset.sigs.k8s.io/). Key fields:

- `replicatedJobs[].name` — use ancestor labels to identify special roles:
  - Label `trainer.kubeflow.org/trainjob-ancestor-step: trainer` on the main training job
  - Label `trainer.kubeflow.org/trainjob-ancestor-step: dataset-initializer` on dataset init jobs
  - Label `trainer.kubeflow.org/trainjob-ancestor-step: model-initializer` on model init jobs

- Container names matter:
  - `node` — the main trainer container (required in jobs with the `trainer` ancestor)
  - `dataset-initializer` — required in jobs with the `dataset-initializer` ancestor
  - `model-initializer` — required in jobs with the `model-initializer` ancestor

- Jobs with ancestor labels must have `replicas: 1`

### 4. Create the manifest file

Place the manifest in the appropriate location:

- **Pre-built runtimes**: `manifests/base/runtimes/`
  - General runtimes: `manifests/base/runtimes/<name>.yaml`
  - TorchTune fine-tuning: `manifests/base/runtimes/torchtune/<model>/<variant>.yaml`
  - Data-cache enabled: `manifests/base/runtimes/data-cache/`

Reference existing runtimes:
- `manifests/base/runtimes/torch_distributed.yaml` — basic PyTorch distributed
- `manifests/base/runtimes/deepspeed_distributed.yaml` — DeepSpeed with MPI
- `manifests/base/runtimes/mlx_distributed.yaml` — Apple MLX
- `manifests/base/runtimes/torchtune/llama3_2/llama3_2_1B.yaml` — TorchTune fine-tuning

### 5. Register in kustomization

Add the new manifest to `manifests/base/runtimes/kustomization.yaml`:
```yaml
resources:
  - <your-new-runtime>.yaml
```

### 6. Add to RHOAI overlay (if applicable)

For RHOAI-specific runtimes, also add to `manifests/rhoai/runtimes/`:
- Create the runtime YAML there
- Update `manifests/rhoai/runtimes/kustomization.yaml`

### 7. Add framework plugin support (if new framework)

If the runtime uses a new ML framework not yet supported:

1. Create a new plugin in `pkg/runtime/framework/plugins/<framework>/`
2. Implement the required plugin interfaces (see `pkg/runtime/framework/interface.go`):
   - `EnforceMLPolicyPlugin` — to inject framework-specific env vars and config
   - `CustomValidationPlugin` — to validate framework-specific fields
   - `ComponentBuilderPlugin` — if the framework needs extra K8s resources (e.g., MPI needs SSH secrets)
3. Register the plugin in `pkg/runtime/framework/plugins/registry.go`
4. Add the framework's ML policy type to `pkg/apis/trainer/v1alpha1/trainingruntime_types.go`

### 8. Validate

- Ensure the runtime passes webhook validation:
  - Jobs with ancestor labels have `replicas: 1`
  - Required containers are present for each ancestor type
  - Only one of `torch` or `mpi` is set in `mlPolicy`
- Test with a TrainJob that references the new runtime
- Run `make manifests` to regenerate if any Go types were changed

### 9. Add Helm chart support (optional)

If the runtime should be installable via Helm:
- Add a template or static file in `charts/kubeflow-trainer/`
- Consider making it toggleable via `values.yaml`

## Example: minimal PyTorch distributed runtime

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: torch-distributed-custom
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  template:
    spec:
      replicatedJobs:
        - name: trainer
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: trainer
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: docker.io/your-org/your-trainer:latest
                      env:
                        - name: TRAINING_SCRIPT
                          value: train.py
                      resources:
                        requests:
                          nvidia.com/gpu: "1"
                  restartPolicy: OnFailure
```