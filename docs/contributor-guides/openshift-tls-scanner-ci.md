# OpenShift TLS and PQC Presubmit

This guide applies to the [`opendatahub-io/trainer`](https://github.com/opendatahub-io/trainer)
repository. It describes the OpenShift Prow presubmit that deploys a pull request image on an
OpenShift cluster, runs the Trainer end-to-end tests, and scans the running workloads for TLS and
post-quantum cryptography (PQC) compatibility.

The Prow configuration is maintained in
[`openshift/release`](https://github.com/openshift/release). The files in that repository are the
source of truth; this page explains their behavior and the contributor workflow.

## Why this check exists

TLS behavior depends on the deployed workload, its certificates, and the protocols exposed by its
running endpoints. Static source checks alone cannot validate that behavior. The Prow job therefore
deploys the image built from the pull request and scans the resulting pods before the change can
merge.

Trainer has three complementary checks:

- [TLS Config Lint](https://github.com/opendatahub-io/trainer/blob/main/.github/workflows/tls-lint.yml)
  finds unsafe TLS configuration.
- [Semgrep TLS Rules](https://github.com/opendatahub-io/trainer/blob/main/.github/workflows/semgrep-tls.yml)
  finds insecure TLS code patterns.
- `ci/prow/tls-scan-pqc` probes the TLS endpoints of running pods on an OpenShift cluster.

The runtime scanner is not an image vulnerability or filesystem scanner.

## Presubmit flow

For a pull request against `main`, Prow performs the following steps:

1. Build `cmd/trainer-controller-manager/Dockerfile.odh` as the `trainer` test image.
2. Claim an OpenShift 4.20 cluster.
3. Deploy `manifests/overlays/manager` in `kubeflow-system`.
4. Replace the manager Deployment image with the exact pull request image supplied through
   `TRAINER_IMAGE`, verify the Deployment spec, and wait for the rollout.
5. Deploy `manifests/overlays/runtimes` and run `make test-e2e`.
6. Run the shared `tls-scanner-run` step against pods in `kubeflow-system` with `PQC_CHECK=true`.
7. Fail the presubmit if deployment, E2E testing, or the scanner fails.

In PQC mode, the shared scanner checks TLS 1.3 and ML-KEM (`mlkem` or `mlkem25519`) support for
the discovered TLS targets. Scanner output is published with the Prow job artifacts, including the
JSON, CSV, JUnit, and scanner log files.

```text
Trainer PR
  -> build PR image
  -> claim OCP cluster
  -> deploy and verify PR image
  -> run E2E tests
  -> scan kubeflow-system pods
  -> pass or block the PR
```

## Configuration locations

- [Trainer source configuration](https://github.com/openshift/release/blob/main/ci-operator/config/opendatahub-io/trainer/opendatahub-io-trainer-main.yaml)
- [Generated Trainer presubmits](https://github.com/openshift/release/blob/main/ci-operator/jobs/opendatahub-io/trainer/opendatahub-io-trainer-main-presubmits.yaml)
- [Shared TLS scanner step](https://github.com/openshift/release/blob/main/ci-operator/step-registry/tls/scanner/run/tls-scanner-run-ref.yaml)

The source configuration defines the image build, cluster claim, deployment and E2E commands,
scanner namespace, and PQC mode. The generated presubmit file must not be edited independently.

## Running and troubleshooting the job

The job normally runs automatically on each pull request. To rerun it, comment on the pull request:

```text
/test tls-scan-pqc
```

Use the Prow job page to identify the failed phase:

- **Image build:** inspect the `ci/prow/images` job and `Dockerfile.odh` output.
- **Deployment:** confirm the manager Deployment uses `TRAINER_IMAGE` and completes its rollout.
- **E2E:** inspect the `make test-e2e` output before investigating the scanner.
- **Scanner:** inspect the TLS scanner log and JSON or JUnit findings in the job artifacts.

The complete presubmit depends on an OpenShift cluster claim and is intended to run in Prow. The
existing Kind E2E suite remains useful for local functional testing, but it does not reproduce the
OpenShift TLS/PQC scan.

## Updating the presubmit

Make Prow changes in a fork of `openshift/release`:

1. Edit the Trainer source configuration under
   `ci-operator/config/opendatahub-io/trainer/`.
2. Run `make jobs` from the `openshift/release` repository root.
3. Review both the source configuration and generated presubmit changes.
4. Run the repository validation checks and create a signed commit.
5. Open a pull request against `openshift/release` and verify its configuration checks.

When changing the deployment flow, keep these invariants:

- Deploy the image built from the pull request, not the image pinned by an overlay.
- Verify the exact image reference before running E2E tests or the scanner.
- Wait until every namespace being scanned has its expected TLS workloads ready.
- Keep `SCAN_NAMESPACE` limited to namespaces owned by the component.

## Trainer Operator during migration

[`opendatahub-io/trainer-operator`](https://github.com/opendatahub-io/trainer-operator) currently
has a separate Prow presubmit because it builds and deploys a different operator image. The Trainer
presubmit does not build or scan that operator image. The separate job installs its prerequisites,
runs the operator's OCP E2E suite, restores the Trainer custom resource removed during test cleanup,
and scans `trainer-operator-system` and `opendatahub`.

Its configuration is maintained in the
[`trainer-operator` section of `openshift/release`](https://github.com/openshift/release/blob/main/ci-operator/config/opendatahub-io/trainer-operator/opendatahub-io-trainer-operator-main.yaml).
Keep the two jobs independent until the operator migration into the Trainer repository is complete;
then consolidate the jobs only after the migrated E2E flow deploys and verifies both workloads.
