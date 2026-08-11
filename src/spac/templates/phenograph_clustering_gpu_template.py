"""
Galaxy GPU template for PhenoGraph clustering.

Calls ``spac.transform.clustering.phenograph.gpu.grapheno.phenograph_gpu``
inside the GPU Docker container (``nciccbr/spac-gpu``). Supports profiling
mode, which runs Leiden at multiple resolutions sharing the same KNN and
Jaccard computation (a grapheno feature exposed via the parquet cache in
the job work directory).

Usage
-----
>>> from spac.templates.phenograph_clustering_gpu_template import run_from_json
>>> run_from_json("examples/phenograph_clustering_gpu_params.json")
"""
import sys
from pathlib import Path
from typing import Any, Dict, Union

sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transform.clustering.phenograph import phenograph_gpu
from spac.templates.template_utils import (
    load_input,
    parse_params,
    save_results,
    text_to_value,
)


def _resolutions(params):
    profiling = bool(params.get("Profiling", False))
    res_list = params.get("Resolution_List") or []
    res_param = params.get("Resolution_Parameter", 1.0)

    if profiling:
        if not res_list:
            raise ValueError(
                "Profiling=True but Resolution_List is empty."
            )
        return True, [float(r) for r in res_list]
    return False, [float(res_param)]


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
):
    """
    Execute GPU PhenoGraph Clustering from Galaxy JSON parameters.
    """
    if phenograph_gpu is None:
        raise RuntimeError(
            "GPU backend not available. Ensure RAPIDS (cuml, cugraph) is "
            "installed and this template is running inside the GPU Docker "
            "image (nciccbr/spac-gpu)."
        )

    params = parse_params(json_path)

    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    if "outputs" not in params:
        params["outputs"] = {
            "analysis": {"type": "file", "name": "output.pickle"}
        }

    # Load AnnData
    adata = load_input(params["Upstream_Analysis"])

    layer = params.get("Table_to_Process", "Original")
    if layer == "Original":
        layer = None

    k = int(params.get("K_Nearest_Neighbors", 30))
    seed = int(params.get("Seed", 42))
    output_name = params.get("Output_Annotation_Name", "phenograph")
    n_iterations = int(params.get("Number_of_Iterations", 100))

    profiling, resolutions = _resolutions(params)

    print("Before PhenoGraph GPU Clustering:\n", adata)
    print(f"Mode: {'profiling' if profiling else 'single'}, resolutions={resolutions}")

    # In profiling mode, the first resolution's call populates the parquet cache
    # (KNN + Jaccard); subsequent calls only re-run Leiden. The cache lives in
    # the process cwd, which Galaxy sets to the per-job work directory.
    for res in resolutions:
        col_name = (
            f"{output_name}-{k}-{res}" if profiling else output_name
        )
        phenograph_gpu(
            adata=adata,
            features=None,
            layer=layer,
            k=k,
            seed=seed,
            resolution_parameter=res,
            n_iterations=n_iterations,
            output_annotation=col_name,
        )
        n_clusters = int(adata.obs[col_name].nunique())
        print(f"  {col_name}: {n_clusters} clusters")

    # Summary column for provenance (single-resolution mode uses output_name;
    # profiling mode uses the last resolution as a reference).
    summary_col = (
        f"{output_name}-{k}-{resolutions[-1]}" if profiling else output_name
    )
    print(f'\nCount of cells in "{summary_col}":')
    print(adata.obs[summary_col].value_counts())
    print("\nAfter PhenoGraph GPU Clustering:\n", adata)

    if save_to_disk:
        saved_files = save_results(
            results={"analysis": adata},
            params=params,
            output_base_dir=output_dir,
        )
        print(
            f"PhenoGraph GPU Clustering completed -> {saved_files['analysis']}"
        )
        return saved_files

    return adata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python phenograph_clustering_gpu_template.py "
            "<params.json> [output_dir]",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    result = run_from_json(json_path=sys.argv[1], output_dir=output_dir)

    if isinstance(result, dict):
        print("\nOutput files:")
        for name, path in result.items():
            print(f"  {name}: {path}")
    else:
        print("\nReturned AnnData object")
