#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=TVB

pushd "${ROOT_DIR}/python"

# Setup conda
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Handle active env safely
if [ -n "${CONDA_PREFIX}" ]; then
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
        conda deactivate
    fi
fi

# Safely remove env if exists
conda env remove -n "${ENV_NAME}" 2>/dev/null || true

# Create env
conda env create -f ./rlgpu_conda_env.yml || {
    echo "*** Failed to create env"
    exit 1
}

# Activate
conda activate "${ENV_NAME}" || {
    echo "*** Failed to activate env"
    exit 1
}

# Verify activation
if [ -z "${CONDA_PREFIX}" ] || [ "$(basename ${CONDA_PREFIX})" != "${ENV_NAME}" ]; then
    echo "*** Env is not active, aborting"
    exit 1
fi

# Install package
pip install -e .

popd

echo "SUCCESS"
