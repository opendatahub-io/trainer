# RHOAI-specific installation targets.
#
# This file is intentionally separate from the upstream Makefile. Use it with:
#
#   make -f Makefile.rhai deploy-rhoai

include Makefile

# Make's SHELL variable accepts only the executable path. Override the upstream
# value here so the RHAI recipes can use Bash options reliably.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -ec
.ONESHELL:

# Kubernetes CLI tool (kubectl or oc).
KUBECTL ?= $(shell command -v oc 2>/dev/null || command -v kubectl 2>/dev/null)
RHOAI_MANIFESTS_DIR ?= $(PROJECT_DIR)/manifests/rhoai
RHAI_TEST_MANIFESTS_DIR ?= $(PROJECT_DIR)/odh_utils/manifests
TRAINER_IMAGE ?=
TRAINER_OPERATOR_RUNTIME_REF ?= main
TRAINER_OPERATOR_RUNTIME_URL ?= https://github.com/opendatahub-io/trainer-operator//manifests/runtimes?ref=$(TRAINER_OPERATOR_RUNTIME_REF)
ARTIFACT_DIR ?= $(PROJECT_DIR)/artifacts
UPSTREAM_JUNIT_REPORT ?= upstream-trainer-junit.xml
DW_SOURCE_DIR ?= $(PROJECT_DIR)/.test-data/distributed-workloads
DW_SOURCE_URL ?= https://github.com/opendatahub-io/distributed-workloads.git
DW_SOURCE_REF ?= main
DW_TEST_LOG ?= $(ARTIFACT_DIR)/distributed-workloads-smoke.log
DW_JUNIT_FILE ?= $(ARTIFACT_DIR)/distributed-workloads-smoke.xml
JUNIT_REPORT ?= $(ARTIFACT_DIR)/junit.xml
GOTESTSUM ?= gotestsum

.PHONY: check-kubectl
check-kubectl:
	@if [[ -z "$(KUBECTL)" ]]; then
		echo "Error: Neither 'oc' nor 'kubectl' found in PATH"
		exit 1
	fi

.PHONY: install-cert-manager
install-cert-manager: check-kubectl ## Install the OpenShift cert-manager operator.
	@echo "Installing cert-manager operator..."
	$(KUBECTL) apply -f $(RHAI_TEST_MANIFESTS_DIR)/cert-manager-operator.yaml
	echo "Waiting for cert-manager CSV to be created by OLM..."
	for i in $$(seq 1 60); do
		CSV=$$($(KUBECTL) get csv -n cert-manager-operator \
			-l operators.coreos.com/openshift-cert-manager-operator.cert-manager-operator \
			-o name 2>/dev/null | head -1)
		if [[ -n "$$CSV" ]]; then
			echo "CSV found: $$CSV"
			break
		fi
		echo "  attempt $$i/60, retrying in 10s..."
		sleep 10
	done
	$(KUBECTL) wait --for=jsonpath='{.status.phase}'=Succeeded csv \
		-n cert-manager-operator \
		-l operators.coreos.com/openshift-cert-manager-operator.cert-manager-operator \
		--timeout=300s
	echo "Waiting for cert-manager deployment to be created by the operator..."
	for i in $$(seq 1 30); do
		if $(KUBECTL) get deployment cert-manager -n cert-manager >/dev/null 2>&1; then
			break
		fi
		echo "  attempt $$i/30, retrying in 10s..."
		sleep 10
	done
	$(KUBECTL) wait --for=condition=available deployment/cert-manager \
		-n cert-manager --timeout=300s

.PHONY: install-jobset-operator
install-jobset-operator: install-cert-manager ## Install and enable the OpenShift JobSet operator.
	@echo "Installing JobSet operator..."
	$(KUBECTL) apply -f $(RHAI_TEST_MANIFESTS_DIR)/jobset-operator.yaml
	echo "Waiting for JobSet CSV to be created by OLM..."
	for i in $$(seq 1 60); do
		CSV=$$($(KUBECTL) get csv -n openshift-jobset-operator \
			-l operators.coreos.com/job-set.openshift-jobset-operator \
			-o name 2>/dev/null | head -1)
		if [[ -n "$$CSV" ]]; then
			echo "CSV found: $$CSV"
			break
		fi
		echo "  attempt $$i/60, retrying in 10s..."
		sleep 10
	done
	$(KUBECTL) wait --for=jsonpath='{.status.phase}'=Succeeded csv \
		-n openshift-jobset-operator \
		-l operators.coreos.com/job-set.openshift-jobset-operator \
		--timeout=300s
	$(KUBECTL) apply -f $(RHAI_TEST_MANIFESTS_DIR)/jobset-operator-cluster.yaml
	echo "Waiting for jobsets CRD to be created by the operator..."
	for i in $$(seq 1 30); do
		if $(KUBECTL) get crd jobsets.jobset.x-k8s.io >/dev/null 2>&1; then
			break
		fi
		echo "  attempt $$i/30, retrying in 10s..."
		sleep 10
	done
	$(KUBECTL) wait --for=condition=established crd/jobsets.jobset.x-k8s.io --timeout=300s
	$(KUBECTL) wait --for=condition=Available jobsetoperator/cluster --timeout=300s

.PHONY: deploy-rhoai
deploy-rhoai: install-jobset-operator ## Install Trainer using RHOAI manifests with kustomize.
	if [[ -n "$(TRAINER_IMAGE)" ]]; then
		echo "Using Trainer image: $(TRAINER_IMAGE)"
		$(SED) -i "s|odh-kubeflow-trainer-controller-image=.*|odh-kubeflow-trainer-controller-image=$(TRAINER_IMAGE)|" "$(RHOAI_MANIFESTS_DIR)/params.env"
	fi
	@echo "Deploying RHOAI Training Operator to namespace: opendatahub"
	$(KUBECTL) create namespace opendatahub --dry-run=client -o yaml | $(KUBECTL) apply -f - >/dev/null 2>&1
	echo "Applying CRDs first..."
	$(KUBECTL) apply --server-side=true -k $(RHOAI_MANIFESTS_DIR)/../base/crds
	echo "Waiting for CRDs to be established..."
	$(KUBECTL) wait --for condition=established --timeout=60s \
		crd/clustertrainingruntimes.trainer.kubeflow.org \
		crd/trainingruntimes.trainer.kubeflow.org \
		crd/trainjobs.trainer.kubeflow.org
	echo "Applying operator and resources..."
	$(KUBECTL) apply -k $(RHOAI_MANIFESTS_DIR) --server-side=true --force-conflicts
	echo "Waiting for deployment to be ready..."
	$(KUBECTL) wait --for=condition=available --timeout=300s \
		deployment/kubeflow-trainer-controller-manager -n opendatahub 2>/dev/null
	echo "RHOAI Training Operator deployed successfully!"

.PHONY: install-test-runtimes
install-test-runtimes: check-kubectl ## Install Trainer runtimes required by the integration tests.
	@echo "Installing midstream Trainer runtimes..."
	$(KUBECTL) apply --server-side -k '$(TRAINER_OPERATOR_RUNTIME_URL)'
	echo "Installing upstream E2E-only runtimes..."
	$(KUBECTL) apply --server-side \
		-f manifests/base/runtimes/jax_distributed.yaml \
		-f manifests/base/runtimes/xgboost_distributed.yaml \
		-f examples/flux/flux-runtime.yaml

.PHONY: test-e2e-rhai
test-e2e-rhai: ginkgo ## Run the upstream Trainer RHAI E2E suite.
	@mkdir -p "$(ARTIFACT_DIR)"
	# TODO: Enable 'runtime status server instrumentation' once the status server is out of the alpha feature gate.
	$(GINKGO) -v --skip='runtime status server instrumentation' \
		--junit-report="$(UPSTREAM_JUNIT_REPORT)" \
		--output-dir="$(ARTIFACT_DIR)" ./test/e2e/...

.PHONY: test-dw-smoke
test-dw-smoke: ## Run the distributed-workloads Trainer smoke suite.
	@test -d "$(DW_SOURCE_DIR)/tests/trainer" || { \
		echo "Trainer smoke tests not found in $(DW_SOURCE_DIR)/tests/trainer"; \
		exit 1; \
	}
	@mkdir -p "$(ARTIFACT_DIR)"
	cd "$(DW_SOURCE_DIR)"
	# This pipeline provisions a bare OpenShift cluster with upstream Trainer,
	# not a full RHOAI/ODH installation. Keep these RHOAI-specific tests skipped:
	# - TestKubeflowTrainerSmoke requires the DSCI resource and rhods-operator.
	# - TestDefaultClusterTrainingRuntimes validates RHOAI/ODH image packaging
	#   through rhods-operator and its expected runtime image set.
	# TestOpenMPICudaClusterTrainingRuntime is skipped in the test source because
	# of RHOAIENG-61966.
	TEST_TIER=Smoke $(GOTESTSUM) \
		--format standard-verbose \
		--junitfile "$(DW_JUNIT_FILE)" \
		--junitfile-testsuite-name short \
		--junitfile-testcase-classname relative \
		--junitfile-hide-skipped-tests \
		--junitfile-project-name=trainer \
		-- -count=1 -timeout 20m \
		-skip='TestKubeflowTrainerSmoke|TestDefaultClusterTrainingRuntimes' \
		./tests/trainer/ 2>&1 | tee "$(DW_TEST_LOG)"

.PHONY: test-integration-rhai
test-integration-rhai: ## Run the complete RHOAI Trainer integration test suite.
	@mkdir -p "$(ARTIFACT_DIR)" "$$(dirname "$(JUNIT_REPORT)")"
	write_empty_junit() {
		printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?><testsuites tests="0" failures="0" errors="0" skipped="0" time="0"></testsuites>' > "$(JUNIT_REPORT)"
	}
	trap 'if [[ ! -s "$(JUNIT_REPORT)" ]]; then write_empty_junit; fi' EXIT
	write_empty_junit
	command -v junitparser >/dev/null 2>&1 || { echo "Error: junitparser not found in PATH"; exit 1; }

	setup_passed=true
	$(MAKE) -f Makefile.rhai deploy-rhoai \
		TRAINER_IMAGE="$(TRAINER_IMAGE)" \
		|| setup_passed=false
	if [[ "$$setup_passed" = true ]]; then
		$(MAKE) -f Makefile.rhai install-test-runtimes || setup_passed=false
	fi
	if [[ "$$setup_passed" != true ]]; then
		echo "Trainer integration test setup failed"
		exit 1
	fi

	upstream_tests_passed=true
	$(MAKE) -f Makefile.rhai test-e2e-rhai \
		ARTIFACT_DIR="$(ARTIFACT_DIR)" \
		|| upstream_tests_passed=false

	dw_tests_passed=true
	if [[ ! -d "$(DW_SOURCE_DIR)/tests/trainer" ]]; then
		if [[ -e "$(DW_SOURCE_DIR)" ]]; then
			echo "Distributed-workloads source directory exists but does not contain Trainer tests: $(DW_SOURCE_DIR)"
			dw_tests_passed=false
		else
			mkdir -p "$$(dirname "$(DW_SOURCE_DIR)")"
			git clone --depth 1 --branch "$(DW_SOURCE_REF)" "$(DW_SOURCE_URL)" "$(DW_SOURCE_DIR)" \
				|| dw_tests_passed=false
		fi
	fi
	if [[ "$$dw_tests_passed" = true ]]; then
		$(MAKE) -f Makefile.rhai test-dw-smoke \
			DW_SOURCE_DIR="$(DW_SOURCE_DIR)" \
			ARTIFACT_DIR="$(ARTIFACT_DIR)" \
			DW_TEST_LOG="$(DW_TEST_LOG)" \
			DW_JUNIT_FILE="$(DW_JUNIT_FILE)" \
			|| dw_tests_passed=false
		# TestMain in the DW Trainer package can exit successfully when a test
		# fails, so verify the generated JUnit testcases as well.
		if [[ ! -s "$(DW_JUNIT_FILE)" ]] || ! junitparser verify "$(DW_JUNIT_FILE)"; then
			dw_tests_passed=false
		fi
	fi

	junit_reports=()
	for report in \
		"$(ARTIFACT_DIR)/$(UPSTREAM_JUNIT_REPORT)" \
		"$(DW_JUNIT_FILE)"; do
		if [[ -s "$$report" ]]; then
			junit_reports+=("$$report")
		fi
	done
	if [[ $${#junit_reports[@]} -eq 0 ]]; then
		echo "No JUnit reports were generated"
		exit 1
	fi
	junitparser merge "$${junit_reports[@]}" "$(JUNIT_REPORT)"

	if [[ "$$upstream_tests_passed" != true || "$$dw_tests_passed" != true ]]; then
		echo "Trainer integration tests failed"
		exit 1
	fi
	echo "Trainer integration tests passed"

.PHONY: undeploy-rhoai
undeploy-rhoai: check-kubectl ## Undeploy Trainer using RHOAI manifests.
	@echo "Undeploying RHOAI Training Operator from namespace: opendatahub"
	echo "Deleting all TrainJob and TrainingRuntime CRs..."
	$(KUBECTL) delete trainjobs.trainer.kubeflow.org --all --all-namespaces --ignore-not-found=true --timeout=60s
	$(KUBECTL) delete trainingruntimes.trainer.kubeflow.org --all --all-namespaces --ignore-not-found=true --timeout=60s
	$(KUBECTL) delete clustertrainingruntimes.trainer.kubeflow.org --all --ignore-not-found=true --timeout=60s
	echo "Deleting operator and resources..."
	$(KUBECTL) delete -k $(RHOAI_MANIFESTS_DIR) --ignore-not-found=true
	echo "RHOAI Training Operator undeployed successfully!"
