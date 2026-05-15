#!/usr/bin/env bash
set -e

GITHUB_URL="https://github.com"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST_MANIFESTS_DIR="${1:-${REPO_ROOT}/opt/manifests}"

# Format: "repo-org:repo-name:ref-name:source-folder"
ODH_MANIFESTS="opendatahub-io:trainer:stable:manifests"
RHOAI_MANIFESTS="red-hat-data-services:trainer:rhoai-3.5-ea.1:manifests"

if [ "${ODH_PLATFORM_TYPE:-OpenDataHub}" = "OpenDataHub" ]; then
    echo "Cloning manifests for ODH"
    COMPONENT_MANIFESTS="${ODH_MANIFESTS}"
else
    echo "Cloning manifests for RHOAI"
    COMPONENT_MANIFESTS="${RHOAI_MANIFESTS}"
fi

TMP_DIR=$(mktemp -d -t "trainer-manifests.XXXXXXXXXX")
trap '{ rm -rf -- "$TMP_DIR"; }' EXIT

function try_fetch_ref()
{
    local repo=$1
    local ref_type=$2
    local ref=$3
    local git_ref="refs/$ref_type/$ref"

    if git ls-remote --exit-code "$repo" "$git_ref" &>/dev/null; then
        if git fetch -q --depth 1 "$repo" "$git_ref" && git reset -q --hard FETCH_HEAD; then
            return 0
        fi
    fi
    return 1
}

function git_fetch_ref()
{
    local repo=$1
    local ref=$2
    local dir=$3

    mkdir -p "$dir"
    pushd "$dir" &>/dev/null
    git init -q

    if [[ $ref =~ ^([a-zA-Z0-9_./-]+)@([a-f0-9]{7,40})$ ]]; then
        local commit_sha="${BASH_REMATCH[2]}"
        git remote add origin "$repo"
        if ! git fetch --depth 1 -q origin "$commit_sha"; then
            echo "ERROR: Failed to fetch from repository $repo"
            popd &>/dev/null
            return 1
        fi
        if ! git reset -q --hard "$commit_sha" 2>/dev/null; then
            echo "ERROR: Commit SHA $commit_sha not found in repository $repo"
            popd &>/dev/null
            return 1
        fi
    else
        if try_fetch_ref "$repo" "tags" "$ref" || try_fetch_ref "$repo" "heads" "$ref"; then
            :
        else
            echo "ERROR: '$ref' is not a valid branch, tag, or commit SHA in repository $repo"
            popd &>/dev/null
            return 1
        fi
    fi

    popd &>/dev/null
}

IFS=':' read -r -a repo_info <<< "${COMPONENT_MANIFESTS}"
repo_org="${repo_info[0]}"
repo_name="${repo_info[1]}"
repo_ref="${repo_info[2]}"
source_path="${repo_info[3]}"

echo -e "\033[32mCloning trainer manifests:\033[0m ${COMPONENT_MANIFESTS}"

repo_url="${GITHUB_URL}/${repo_org}/${repo_name}"
repo_dir="${TMP_DIR}/trainer"

if ! git_fetch_ref "${repo_url}" "${repo_ref}" "${repo_dir}"; then
    echo "ERROR: Failed to fetch ref '${repo_ref}' from '${repo_url}'"
    exit 1
fi

mkdir -p "${DST_MANIFESTS_DIR}"
cp -rf "${repo_dir}/${source_path}"/* "${DST_MANIFESTS_DIR}/"

echo "  trainer: $(find "${DST_MANIFESTS_DIR}" -type f | wc -l) files"
echo "Done."
