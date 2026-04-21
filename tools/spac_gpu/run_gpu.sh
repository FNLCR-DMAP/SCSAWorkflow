#!/bin/bash
# /opt/spac-gpu/run_gpu.sh
# -----------------------------------------------------------------------------
# Orchestrator for PhenoGraph GPU clustering inside nciccbr/spac-gpu.
#
# This script exists because RAPIDS 22.08 pins numpy/pandas versions
# incompatible with anndata, so the container uses two conda envs:
#
#   preprocess env: anndata + pandas 1.5 + numpy 1.26
#   rapids env:     cuml 22.08 + cugraph 22.08
#
# Three stages, each in the right env:
#
#   Stage 1 (preprocess): pickle -> adata.pickle + features metadata
#                         (uses the Python template end-to-end minus clustering)
#   Stage 2 (rapids):     call grapheno via run_grapheno.py
#                         (writes cluster_result_<res>.csv per resolution)
#   Stage 3 (preprocess): merge labels back into adata, save output.pickle
#
# NOTE: When migrating to RAPIDS 24.12, delete this script and have the
# Galaxy XML call the template directly in a single conda env.
# -----------------------------------------------------------------------------
# Usage:
#   bash /opt/spac-gpu/run_gpu.sh <cleaned_params.json>
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <cleaned_params.json>" >&2
    exit 1
fi

PARAMS_JSON="$(realpath "$1")"
WORK_DIR="$(pwd)"

if [[ ! -f "$PARAMS_JSON" ]]; then
    echo "ERROR: params file not found: $PARAMS_JSON" >&2
    exit 1
fi

echo "=============================================================="
echo " SPAC PhenoGraph GPU Clustering (grapheno / RAPIDS 22.08)"
echo " Work dir : $WORK_DIR"
echo " Params   : $PARAMS_JSON"
echo " GPU      :"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo " (nvidia-smi not available)"
echo "=============================================================="

source /opt/conda/etc/profile.d/conda.sh

# -----------------------------------------------------------------------------
# STAGE 1 — preprocess env: load pickle, extract features matrix to disk
# -----------------------------------------------------------------------------
echo
echo "[1/3] Loading pickle, extracting features (preprocess env)..."
conda activate preprocess

python - <<'PYEOF' "$PARAMS_JSON" "$WORK_DIR"
import json, os, pickle, sys
import numpy as np

import anndata  # noqa  (assert available)
from spac.templates.template_utils import load_input, parse_params
from spac.transform.clustering.phenograph import prepare_features

params_path = sys.argv[1]
work_dir = sys.argv[2]

params = parse_params(params_path)
input_path = params["Upstream_Analysis"]
layer = params.get("Table_to_Process", "Original")
if layer == "Original":
    layer = None

print(f"  Loading: {input_path}", flush=True)
adata = load_input(input_path)
print(f"  AnnData: {adata.n_obs:,} obs x {adata.n_vars} vars", flush=True)

data, feature_names = prepare_features(adata, layer=layer)
print(
    f"  Features matrix: {data.shape[0]:,} x {data.shape[1]} "
    f"(layer='{layer or 'X'}')",
    flush=True,
)

# Write intermediates that stage 2 (rapids env) will consume
np.save(os.path.join(work_dir, "features.npy"), data)
with open(os.path.join(work_dir, "features_meta.json"), "w") as f:
    json.dump({"feature_names": feature_names, "n_obs": int(adata.n_obs)}, f)
with open(os.path.join(work_dir, "adata.pickle"), "wb") as f:
    pickle.dump(adata, f)

print("  Stage 1 complete.", flush=True)
PYEOF

# -----------------------------------------------------------------------------
# STAGE 2 — rapids env: run grapheno clustering, one or more resolutions
# -----------------------------------------------------------------------------
echo
echo "[2/3] Running grapheno clustering (rapids env)..."
conda activate rapids

python -u /opt/spac-gpu/run_grapheno.py "$PARAMS_JSON" "$WORK_DIR"

# -----------------------------------------------------------------------------
# STAGE 3 — preprocess env: merge labels back, save output.pickle
# -----------------------------------------------------------------------------
echo
echo "[3/3] Merging labels into AnnData (preprocess env)..."
conda activate preprocess

python -u /opt/spac-gpu/merge_labels.py "$PARAMS_JSON" "$WORK_DIR"

echo
echo "=============================================================="
echo " PhenoGraph GPU clustering completed."
echo "=============================================================="
