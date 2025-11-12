# RHAI (Red Hat AI) Extensions

This directory contains RHAI-specific extensions for the Kubeflow Trainer operator.

## Purpose

The `rhai/` package provides midstream-specific features that are not part of upstream Kubeflow:
- **Progression tracking**: Real-time training metrics polling and status updates
- **Custom annotations**: RHAI-specific metadata for training jobs
- **Extended RBAC**: Additional permissions

## Structure

```
pkg/rhai/
├── README.md                       # This file
├── setup.go                        # RHAI feature registration
├── controller/
│   └── progression_controller.go  # Wraps base controller with progression tracking
└── progression/
    ├── progression.go              # Core progression tracking logic
    └── progression_test.go         # Tests for progression tracking
```

## How It Works

### 1. Controller Wrapping

The `ProgressionReconciler` wraps the base `TrainJobReconciler` and adds:
- Metrics polling from training pods
- Progress annotation updates
- Automatic requeuing for ongoing polling

### 2. Progression Tracking

When enabled via annotation `trainer.opendatahub.io/progression-tracking: "enabled"`:
- Controller polls training pod's metrics endpoint (default port: 28080)
- Updates TrainJob annotations with real-time progress
- Captures final metrics on job completion/failure

### 3. Manifest Integration

RHAI-specific manifests in `manifests/rhoai/`:
- `rbac_progression_patch.yaml`: Additional RBAC for pod access
- `manager_config_patch.yaml`: ConfigMap mounting for feature flags

## Enabling RHAI Features

RHAI features are controlled via the `ENABLE_RHAI_FEATURES` environment variable. When enabled, the operator uses a wrapping controller that adds progression tracking to the base upstream functionality.

### Deployment Configuration

**For Kubernetes/OpenShift deployments**, set the environment variable in your deployment manifest:

```yaml
# manifests/rhoai/manager_config_patch.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: manager
        env:
        - name: ENABLE_RHAI_FEATURES
          value: "true"
```

**For local development**, export the variable before running:

```bash
export ENABLE_RHAI_FEATURES=true
go run ./cmd/trainer-controller-manager/main.go
```

## Usage Example

Create a TrainJob with progression tracking:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: pytorch-example
  annotations:
    trainer.opendatahub.io/progression-tracking: "enabled"
    trainer.opendatahub.io/metrics-port: "28080"           # optional, default: 28080
    trainer.opendatahub.io/metrics-poll-interval: "30s"   # optional, default: 30s
spec:
  # ... your training job spec ...
```

The controller will:
1. Poll the primary pod's metrics endpoint every 30s(default) - configurable
2. Update the `trainer.opendatahub.io/trainerStatus` annotation with:
   - Progress percentage
   - Current step/epoch
   - Loss and learning rate
   - Time elapsed/remaining
   - Custom metrics
3. Capture final status when job completes

## Development

### Running Tests

```bash
go test ./pkg/rhai/...
```

### Adding New RHAI Features

1. Create new package under `pkg/rhai/yourfeature/`
2. Add controller wrapper if needed in `pkg/rhai/controller/`
3. Update `pkg/rhai/setup.go` to register the feature
4. Add manifest patches in `manifests/rhoai/`
5. Document in this README

## Maintenance

When rebasing from upstream:
1. Pull upstream changes: `git pull upstream master`
2. Rebase: `git rebase upstream/master`
3. `pkg/rhai/` should auto-merge (no conflicts expected)
4. Review controller integration in `main.go` if base controller changed
5. Run tests: `make test`
