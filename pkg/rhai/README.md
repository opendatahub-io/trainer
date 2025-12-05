# RHAI (Red Hat AI) Extensions

This directory contains RHAI-specific extensions for the Kubeflow Trainer operator.

## Purpose

The `rhai/` package provides midstream-specific features that are not part of upstream Kubeflow:
- **Progression tracking**: Real-time training metrics polling and status updates
- **Custom annotations**: RHAI-specific metadata for training jobs
- **Extended RBAC**: Additional permissions for pod access

## Structure

```
pkg/rhai/
├── constants/          # RHAI-specific constants and annotations
└── progression/        # Core progression tracking logic and tests

pkg/test/e2e/
└── rhai/               # End-to-end tests for rhai functionality
```

## Integration

Progression tracking is integrated into `TrainJobReconciler` with 2 lines:

```go
result, progressionErr := progression.ReconcileProgression(ctx, r.client, log, &trainJob)
return result, errors.Join(err, progressionErr)
```

- All RHAI logic isolated in `pkg/rhai/`
- Enabled per-TrainJob via annotation
- No-op when disabled

## Usage

Enable progression tracking via annotation:

```yaml
metadata:
  annotations:
    trainer.opendatahub.io/progression-tracking: "enabled"
    trainer.opendatahub.io/metrics-port: "28080"           # optional
    trainer.opendatahub.io/metrics-poll-interval: "30s"   # optional
```

Your training container exposes metrics at `http://localhost:28080/metrics`:

```json
{
  "progressPercentage": 45,
  "currentStep": 450,
  "totalSteps": 1000,
  "trainMetrics": {"loss": 0.235}
}
```

Controller updates `trainer.opendatahub.io/trainerStatus` annotation with progress.

User can monitor progress in realtime with training parameters using watch command below :
```
watch -n 2 'kubectl get trainjob <job-name> -n <namespace> -o jsonpath="{.metadata.annotations.trainer\.opendatahub\.io/trainerStatus}" | jq'

```

## Development


Run unit tests with
```bash
go test ./pkg/rhai/...
```

Run e2e tests with:
```bash
make test-e2e-setup-cluster
go test ./test/e2e/rhai... -v -timeout 30m -ginkgo.v -ginkgo.progress
```
