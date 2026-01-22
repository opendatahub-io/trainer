#!/usr/bin/env bash
#
# This script adjusts a running OLM-deployed ODH/RHODS operator to use custom
# component manifests, based on the 'hack/component-dev/README.md'.
#
# It is designed to deploy the 'trainer' component from local manifests.
#
# THIS IS FOR DEVELOPMENT USE ONLY. DO NOT USE IN PRODUCTION.
#
# Prerequisites:
# - 'oc' CLI installed and logged into an OpenShift cluster.
# - 'jq' CLI installed (for JSON parsing).
# - The ODH/RHODS operator must be installed via OLM and in a "Succeeded" state.
#

set -eou pipefail

# --- Configuration ---

# Namespace where the ODH/RHODS operator is deployed
OPERATOR_NAMESPACE="openshift-operators"

# CSV name pattern to search for (supports both ODH and RHODS)
# Will search for CSVs matching these patterns
CSV_NAME_PATTERNS=("rhods-operator" "opendatahub-operator")

# Deployment name within the CSV (will be auto-detected)
OPERATOR_DEPLOYMENT_NAME=""

# --- Component Configuration ---

# Name of the component to override
COMPONENT_NAME="trainer"

# Name for the new PVC
PVC_NAME="${COMPONENT_NAME}-manifests"

# Mount path inside the operator pod. This MUST match the path
# the operator expects for this component's manifests.
MOUNT_PATH="/opt/manifests/${COMPONENT_NAME}"

# --- Source Manifests Configuration ---

# Use local manifests from this project
# Path relative to the script directory
MANIFEST_SOURCE_PATH="../manifests"

# --- End Configuration ---

# --- Prerequisite Check ---
if ! command -v oc &> /dev/null; then
    echo "❌ Error: 'oc' command not found. Please install the OpenShift CLI."
    exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "❌ Error: 'jq' command not found. Please install jq."
    exit 1
fi

echo "✅ Pre-checks passed."
echo "Running component override for: $COMPONENT_NAME"
echo "  Namespace: $OPERATOR_NAMESPACE"
echo "  PVC: $PVC_NAME"
echo "  Mount Path: $MOUNT_PATH"
echo "  Source Path: $MANIFEST_SOURCE_PATH (local)"
echo "---"


# --- Step 1: Detect CSV and Deployment Info ---
echo "---"
echo "➡️ Step 1: Detecting ClusterServiceVersion (CSV) and deployment info..."

# Find the name of the ClusterServiceVersion by trying multiple patterns
CSV_NAME=""
DETECTED_PATTERN=""
for pattern in "${CSV_NAME_PATTERNS[@]}"; do
    CSV_NAME=$(oc get csv -n "$OPERATOR_NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"$pattern\")) | .metadata.name" | head -1)
    if [ -n "$CSV_NAME" ]; then
        echo "  Found CSV: $CSV_NAME"
        DETECTED_PATTERN="$pattern"
        break
    fi
done

if [ -z "$CSV_NAME" ]; then
    echo "❌ Error: Could not find ODH/RHODS operator CSV in namespace '$OPERATOR_NAMESPACE'."
    echo "   Searched for CSVs starting with: ${CSV_NAME_PATTERNS[*]}"
    echo "   Available CSVs:"
    oc get csv -n "$OPERATOR_NAMESPACE" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | sed 's/^/     - /'
    exit 1
fi

# Auto-detect the deployment name from the CSV
OPERATOR_DEPLOYMENT_NAME=$(oc get csv "$CSV_NAME" -n "$OPERATOR_NAMESPACE" -o jsonpath='{.spec.install.spec.deployments[0].name}')
if [ -z "$OPERATOR_DEPLOYMENT_NAME" ]; then
    echo "❌ Error: Could not detect deployment name from CSV '$CSV_NAME'."
    exit 1
fi
echo "  Detected deployment: $OPERATOR_DEPLOYMENT_NAME"

# Determine the deployment namespace based on which operator was detected
# RHODS deploys to redhat-ods-operator, ODH deploys to opendatahub
if [ "$DETECTED_PATTERN" = "rhods-operator" ]; then
    DEPLOYMENT_NAMESPACE="redhat-ods-operator"
elif [ "$DETECTED_PATTERN" = "opendatahub-operator" ]; then
    DEPLOYMENT_NAMESPACE="opendatahub-operators"
else
    echo "❌ Error: Detected pattern '$DETECTED_PATTERN' does not match any known operator pattern."
    echo "   Expected patterns: ${CSV_NAME_PATTERNS[*]}"
    exit 1
fi
echo "  Deployment namespace: $DEPLOYMENT_NAMESPACE"


# --- Step 2: Create PersistentVolumeClaim ---
echo "---"
echo "➡️ Step 2: Creating PersistentVolumeClaim ($PVC_NAME) in namespace $DEPLOYMENT_NAMESPACE..."

# Use 'oc apply' to be idempotent.
cat <<EOF | oc apply -n "$DEPLOYMENT_NAMESPACE" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF
echo "✅ PVC created or already exists."


# --- Step 3: Patch ClusterServiceVersion (CSV) ---
echo "---"
echo "➡️ Step 3: Patching ClusterServiceVersion (CSV)..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Patching CSV: $CSV_NAME in namespace $DEPLOYMENT_NAMESPACE"

if ! oc patch csv "$CSV_NAME" -n "$DEPLOYMENT_NAMESPACE" \
  --type json \
  --patch-file "$SCRIPT_DIR/csv-patch.json"; then
    echo "❌ Error: Failed to patch CSV. Check the patch file and CSV structure."
    exit 1
fi

echo "✅ CSV patched. OLM will now update the operator Deployment."


# --- Step 4: Wait for operator pod readiness ---
echo "---"
echo "➡️ Step 4: Waiting for operator pod to be ready..."
echo "  (This may take a minute as it tears down the old pod and starts a new one)"

# Give OLM time to process the CSV patch and trigger the deployment update
echo "  Waiting for OLM to trigger deployment update..."
sleep 10

# Wait for the deployment to be ready
oc rollout status deployment/"$OPERATOR_DEPLOYMENT_NAME" -n "$DEPLOYMENT_NAMESPACE" --timeout=300s

echo "✅ Operator pod is Ready."


# --- Step 5: Get the operator pod name ---
echo "---"
echo "➡️ Step 5: Retrieving operator pod name..."

# Get the pod selector labels from the deployment
POD_SELECTOR=$(oc get deployment "$OPERATOR_DEPLOYMENT_NAME" -n "$DEPLOYMENT_NAMESPACE" -o jsonpath='{.spec.selector.matchLabels}' 2>/dev/null || true)

if [ -n "$POD_SELECTOR" ]; then
    # Convert JSON matchLabels to label selector format (e.g., "key1=value1,key2=value2")
    LABEL_SELECTOR=$(echo "$POD_SELECTOR" | jq -r 'to_entries | map("\(.key)=\(.value)") | join(",")')
    echo "  Using deployment's pod selector: $LABEL_SELECTOR"
    OPERATOR_POD_NAME=$(oc get pods -n "$DEPLOYMENT_NAMESPACE" -l "$LABEL_SELECTOR" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
fi

if [ -z "$OPERATOR_POD_NAME" ]; then
    echo "❌ Error: Could not find operator pod. Available pods in namespace:"
    oc get pods -n "$DEPLOYMENT_NAMESPACE" -o name | sed 's/^/     - /'
    echo ""
    echo "   Debug: Deployment selector labels:"
    oc get deployment "$OPERATOR_DEPLOYMENT_NAME" -n "$DEPLOYMENT_NAMESPACE" -o jsonpath='{.spec.selector.matchLabels}' | jq '.' || echo "     Could not retrieve deployment selector"
    exit 1
fi

echo "  Found pod: $OPERATOR_POD_NAME"


# --- Step 6: Copy manifests to operator pod ---
echo "---"
echo "➡️ Step 6: Copying manifests to operator pod..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PATH="${SCRIPT_DIR}/${MANIFEST_SOURCE_PATH}"

if [ ! -d "$SOURCE_PATH" ]; then
    echo "❌ Error: Source path '$SOURCE_PATH' does not exist."
    echo "   Expected path: $SOURCE_PATH"
    exit 1
fi

echo "  Copying from: $SOURCE_PATH"
echo "  Copying to: ${DEPLOYMENT_NAMESPACE}/${OPERATOR_POD_NAME}:${MOUNT_PATH}"

if ! oc cp "$SOURCE_PATH/." "${DEPLOYMENT_NAMESPACE}/${OPERATOR_POD_NAME}:${MOUNT_PATH}"; then
    echo "❌ Error: Failed to copy manifests to pod."
    exit 1
fi

echo "✅ Manifests copied successfully."
echo ""
echo "🎉 Component override complete!"
echo "   The operator will now use local manifests from: $SOURCE_PATH"
# --- Step 7: Wait for operator pod to be ready ---
echo "---"
echo "➡️ Step 7: Waiting for operator pod to be ready..."
echo "  (This may take a minute as it tears down the old pod and starts a new one)"

# Wait for the deployment to be ready
oc rollout status deployment/"$OPERATOR_DEPLOYMENT_NAME" -n "$DEPLOYMENT_NAMESPACE" --timeout=300s

echo "✅ Operator pod is Ready."
