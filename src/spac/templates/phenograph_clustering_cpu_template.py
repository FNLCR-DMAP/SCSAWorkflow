"""
Galaxy CPU template for PhenoGraph clustering.

Calls the standalone CPU path in
``spac.transform.clustering.phenograph.cpu.phenograph_cpu``. INDEPENDENT of
the legacy ``spac.transformations.phenograph_clustering`` function — that
continues to serve the legacy ``phenograph_clustering.xml`` tool unchanged.

Usage
-----
>>> from spac.templates.phenograph_clustering_cpu_template import run_from_json
>>> run_from_json("examples/phenograph_clustering_cpu_params.json")
"""
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for SPAC imports (matches legacy template pattern)
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transform.clustering.phenograph import phenograph_cpu
from spac.templates.template_utils import (
    load_input,
    parse_params,
    save_results,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
):
    """
    Execute CPU PhenoGraph Clustering from Galaxy JSON parameters.

    Mirrors the legacy template's run_from_json contract for drop-in
    replacement, minus the dead HPC fields.
    """
    params = parse_params(json_path)

    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    if "outputs" not in params:
        params["outputs"] = {
            "analysis": {"type": "file", "name": "output.pickle"}
        }

    # Load AnnData
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    layer = params.get("Table_to_Process", "Original")
    if layer == "Original":
        layer = None

    k = int(params.get("K_Nearest_Neighbors", 30))
    seed = int(params.get("Seed", 42))
    resolution = float(params.get("Resolution_Parameter", 1.0))
    output_name = params.get("Output_Annotation_Name", "phenograph")
    n_iterations = int(params.get("Number_of_Iterations", 100))

    print("Before PhenoGraph CPU Clustering:\n", adata)

    phenograph_cpu(
        adata=adata,
        features=None,  # use all var index
        layer=layer,
        k=k,
        seed=seed,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        output_annotation=output_name,
    )

    print(f'Count of cells in the output annotation "{output_name}":')
    print(adata.obs[output_name].value_counts())
    print("\nAfter PhenoGraph CPU Clustering:\n", adata)

    if save_to_disk:
        saved_files = save_results(
            results={"analysis": adata},
            params=params,
            output_base_dir=output_dir,
        )
        print(
            f"PhenoGraph CPU Clustering completed -> {saved_files['analysis']}"
        )
        return saved_files

    return adata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python phenograph_clustering_cpu_template.py "
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
