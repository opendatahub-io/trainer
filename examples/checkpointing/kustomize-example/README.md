# TRL Training with Checkpointing - Kustomize Example

This directory contains a Kustomize-based example for TRL training using Trainer V2 with multi-GPU/multi-node support and automated checkpointing.

## 📁 Directory Structure

```
kustomize-example/
├── kustomization.yaml       # Kustomize configuration
├── pvc.yaml                 # PersistentVolumeClaim for shared storage
├── training-runtime.yaml    # TrainingRuntime with initializers
├── trainjob.yaml           # TrainJob definition
├── trl_training.py         # Training script (referenced by ConfigMap)
└── debug-checkpoint-api-pod.yaml  # Debug pod (optional)
```

### 1. Deploy the Example

```bash
# From the training-operator root directory
cd examples/checkpointing/kustomize-example

# Apply using kubectl with kustomize
kubectl apply -k . -n your-namespace

```

### 2. Monitor Training Progress

```bash

# Check training progress (if checkpointing controller is enabled)
# using TrainJob Status
 oc get trainjob trl-demo -n abdhumal-test -o jsonpath='{.status.trainingProgress}' | jq .

#using API
oc get --raw "/apis/trainer.kubeflow.org/v1alpha1/namespaces/abdhumal-test/trainjobs/trl-demo" | jq '.status.trainingProgress'

#using API curl
API_SERVER=$(oc config view --minify -o jsonpath='{.clusters[0].cluster.server}')
TOKEN=$(oc whoami -t)
curl -H "Authorization: Bearer $TOKEN" \
-H "Accept: application/json" \
-k "$API_SERVER/apis/trainer.kubeflow.org/v1alpha1/namespaces/abdhumal-test/trainjobs/trl-demo" \
| jq '.status.trainingProgress'

% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
Dload  Upload   Total   Spent    Left  Speed
100 14904    0 14904    0     0  17150      0 --:--:-- --:--:-- --:--:-- 17150
{
    "checkpointing": {
    "checkpointsCreated": 0,
    "enabled": true,
    "latestCheckpointTime": "2025-09-07T16:06:37Z"
    },
    "epoch": 2,
    "lastUpdateTime": "2025-09-07T16:06:37Z",
    "learningRate": "0.000041",
    "loss": "2.4058",
    "percentComplete": "28.3",
    "step": 354,
    "totalEpochs": 10,
    "totalSteps": 1250
}

```