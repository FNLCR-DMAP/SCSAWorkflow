"""Functions for point patterns spatial statistics."""
from __future__ import annotations

from typing import Union  # noqa: F401
from typing import TYPE_CHECKING
from typing_extensions import Literal

from scanpy import logging as logg
from anndata import AnnData

from numpy.random import default_rng
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _save_data, _assert_spatial_basis, _assert_categorical_obs
from squidpy._constants._constants import RipleyStat
from squidpy._constants._pkg_constants import Key

__all__ = ["ripley"]

# This code is refactored from squidpy (https://github.com/theislab/squidpy)
# Major changes:
# Calculate Ripley L for two phenotypes
# Pass in the precalculated area
# Pass in the number of observations used in the simulation
# https://github.com/theislab/squidpy/blob/main/src/squidpy/external/_ripley.py


#@d.dedent
#@inject_docs(key=Key.obsm.spatial, rp=RipleyStat)
def ripley(
    adata: AnnData,
    cluster_key: str,
    mode: Literal["F", "G", "L"] = "F",
    spatial_key: str = Key.obsm.spatial,
    metric: str = "euclidean",
    n_neigh: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    max_dist: float | None = None,
    n_steps: int = 50,
    support: List[float] | None = None,
    seed: int | None = None,
    area: float | None = None,
    copy: bool = False,
    phenotypes: Tuple[str, str] | None = None
) -> dict[str, pd.DataFrame | NDArrayA]:
    r"""
    Calculate various Ripley's statistics for point processes.

    According to the `'mode'` argument, it calculates one of the following Ripley's statistics:
    `{rp.F.s!r}`, `{rp.G.s!r}` or `{rp.L.s!r}` statistics.

    `{rp.F.s!r}`, `{rp.G.s!r}` are defined as:

    .. math::

        F(t),G(t)=P( d_{{i,j}} \le t )

    Where :math:`d_{{i,j}}` represents:

        - distances to a random Spatial Poisson Point Process for `{rp.F.s!r}`.
        - distances to any other point of the dataset for `{rp.G.s!r}`.

    `{rp.L.s!r}` we first need to compute :math:`K(t)`, which is defined as:

    .. math::

        K(t) = \frac{{1}}{{\lambda}} \sum_{{i \ne j}} \frac{{I(d_{{i,j}}<t)}}{{n}}

    and then we apply a variance-stabilizing transformation:

    .. math::

        L(t) = (\frac{{K(t)}}{{\pi}})^{{1/2}}

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    mode
        Which Ripley's statistic to compute.
    %(spatial_key)s
    metric
        Which metric to use for computing distances.
        For available metrics, check out :class:`sklearn.neighbors.DistanceMetric`.
    n_neigh
        Number of neighbors to consider for the KNN graph.
    n_simulations
        How many simulations to run for computing p-values.
    n_observations
        How many observations to generate for the Spatial Poisson Point Process.
    max_dist
        Maximum distances for the support. If `None`, `max_dist=`:math:`\sqrt{{area \over 2}}`.
    n_steps
        Number of steps for the support.
    support
        list of bins (radiis) for the support. Overrides `max_dist` and `n_steps`.
    phenotypes
        For Ripley L, calculate the function for the cells of these two phenotypes.
    %(seed)s
        The seed for the random number generator.
    %(area)s
        Use passed value for area instead of the area of the convex hull.
    %(copy)s

    Returns
    -------
    %(ripley_stat_returns)s

    References
    ----------
    For reference, check out
    `Wikipedia <https://en.wikipedia.org/wiki/Spatial_descriptive_statistics#Ripley's_K_and_L_functions>`_
    or :cite:`Baddeley2015-lm`.
    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)
    coordinates = adata.obsm[spatial_key]
    clusters = adata.obs[cluster_key].values

    mode = RipleyStat(mode)  # type: ignore[assignment]
    if TYPE_CHECKING:
        assert isinstance(mode, RipleyStat)

    # old squidpy code
    # N = coordinates.shape[#0]
    hull = ConvexHull(coordinates)
    # pass in the area instead of convex hull
    # This is useful when using the same area in multiple ROIs
    if area is None:
        area = hull.volume
    logg.info(f"Area:{area}")
    if max_dist is None:
        max_dist = (area / 2) ** 0.5
    if support is None:
        support = np.linspace(0, max_dist, n_steps)
    else:
        if not isinstance(support, list) or not all(
                isinstance(x, (int, float)) for x in support
        ):
            raise ValueError(f"Support expected to be a list of floats,"
                             f" got {support}.")

        support = np.array(support)
        n_steps = len(support)

    # Check that support is a list of floats

    # prepare labels
    le = LabelEncoder().fit(clusters)
    cluster_idx = le.transform(clusters)

    if phenotypes is None:
        obs_arr = np.empty((le.classes_.shape[0], n_steps))
    else:
        obs_arr = np.empty((1, n_steps))

        # Remove the diagnoal distances from the distances matrix
        remove_diagonal = False
        if phenotypes[0] == phenotypes[1]:
            remove_diagonal = True

    start = logg.info(
        f"Calculating Ripley's {mode} statistic for `{le.classes_.shape[0]}` clusters and `{n_simulations}` simulations"
    )

    if phenotypes is None:

        for i in np.arange(np.max(cluster_idx) + 1):
            coord_c = coordinates[cluster_idx == i, :]
            if mode == RipleyStat.F:
                logg.warning(
                    f"Running the Ripley F simulations for phenotype ID:{i} "
                    f"with n_cells:{n_observations}"
                )
                random = _ppp(hull, n_simulations=1, n_observations=n_observations, seed=seed)
                tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(coord_c)
                distances, _ = tree_c.kneighbors(random, n_neighbors=n_neigh)
                bins, obs_stats = _f_g_function(distances.squeeze(), support)
            elif mode == RipleyStat.G:
                tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(coord_c)
                distances, _ = tree_c.kneighbors(coordinates[cluster_idx != i, :], n_neighbors=n_neigh)
                bins, obs_stats = _f_g_function(distances.squeeze(), support)
            elif mode == RipleyStat.L:

                n_center = n_observations
                n_neighbor = n_observations
                distances = pdist(coord_c, metric=metric)
                bins, obs_stats = _l_function(distances, support,
                n_observations, area)
            else:
                raise NotImplementedError(f"Mode `{mode.s!r}` is not yet implemented.")

            obs_arr[i] = obs_stats
    else:
        if mode == RipleyStat.L:
            center_phenotype = phenotypes[0]
            neighbor_phenotype = phenotypes[1]

            # Index of center and neighbor cells
            center_idx = le.transform([center_phenotype])[0]
            neighbor_idx = le.transform([neighbor_phenotype])[0]

            # Get a boolean series
            center_cell_bool = cluster_idx == center_idx
            neighbor_cell_bool = cluster_idx == neighbor_idx

            n_center = center_cell_bool.sum()
            n_neighbor = neighbor_cell_bool.sum()

            center_coord = coordinates[center_cell_bool, :]
            neighbor_coord = coordinates[neighbor_cell_bool, :]

            distances = cdist(center_coord, neighbor_coord)

            bins, obs_stats = _l_multiple_function(distances,
                                                   support,
                                                   n_center,
                                                   n_neighbor,
                                                   area,
                                                   remove_diagonal)
            obs_arr[0] = obs_stats
        else:
            raise NotImplementedError(f"Mode `{mode.s!r}` is not yet implemented.")

    sims = np.empty((n_simulations, len(bins)))
    pvalues = np.ones((le.classes_.shape[0], len(bins)))
    rng = default_rng(None if seed is None else seed)

    if phenotypes is None:
        logg.warning(f"Running the simulations with n_cells:{n_observations}")
        for i in range(n_simulations):
            random_i = _ppp(
                hull, n_simulations=1,
                n_observations=n_observations, rng=rng, seed=seed)
            if mode == RipleyStat.F:
                tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(random_i)
                distances_i, _ = tree_i.kneighbors(random, n_neighbors=1)
                _, stats_i = _f_g_function(distances_i.squeeze(), support)
            elif mode == RipleyStat.G:
                tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(random_i)
                distances_i, _ = tree_i.kneighbors(coordinates, n_neighbors=1)
                _, stats_i = _f_g_function(distances_i.squeeze(), support)
            elif mode == RipleyStat.L:
                distances_i = pdist(random_i, metric=metric)
                _, stats_i = _l_function(distances_i, support, N, area)
            else:
                raise NotImplementedError(f"Mode `{mode.s!r}` is not yet "
                                          "implemented.")

            # These lines for code runs for every single phenotype
            # to see if the simulation value was greater than the
            # actual value for that phenotype 'j' in obs_arry[j]
            for j in range(obs_arr.shape[0]):
                pvalues[j] += stats_i >= obs_arr[j]
            sims[i] = stats_i

    else:
        for i in range(n_simulations):
            if mode == RipleyStat.L:
                random_i = _ppp(hull,
                                n_simulations=1,
                                n_observations=n_center+n_neighbor,
                                rng=rng,
                                seed=seed)

                # Randomly select the frist n_center cells as center cells
                center_coord = random_i[0:n_center, :]
                if remove_diagonal:
                    neighbor_coord = center_coord
                else:
                    neighbor_coord = random_i[n_center:]
                distances_i = cdist(center_coord, neighbor_coord)
                _, stats_i = _l_multiple_function(distances_i,
                                                  support,
                                                  n_center,
                                                  n_neighbor,
                                                  area,
                                                  remove_diagonal)
                sims[i] = stats_i

            else:
                raise NotImplementedError(f"Mode `{mode.s!r}` is not yet "
                                          "implemented for"
                                          "multipe phenotypes.")

    pvalues /= n_simulations + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    if phenotypes is None:
        obs_df = _reshape_res(obs_arr.T,
                              columns=le.classes_,
                              index=bins,
                              var_name=cluster_key)
    else:
        obs_df = _reshape_res(obs_arr.T,
                              columns=[f"{phenotypes[0]}_{phenotypes[1]}"],
                              index=bins,
                              var_name=cluster_key)

    sims_df = _reshape_res(sims.T,
                           columns=np.arange(n_simulations),
                           index=bins,
                           var_name="simulations")

    res = {
        f"{mode}_stat": obs_df,
        "sims_stat": sims_df,
        "bins": bins,
        "pvalues": pvalues,
        "n_center": n_center,
        "n_neighbor": n_neighbor,
        'area': area
        }

    if TYPE_CHECKING:
        assert isinstance(res, dict)

    if copy:
        logg.info("Finish", time=start)
        return res

    _save_data(adata, attr="uns",
               key=Key.uns.ripley(cluster_key, mode),
               data=res,
               time=start)


def _reshape_res(results: NDArrayA, columns: NDArrayA | list[str], index: NDArrayA, var_name: str) -> pd.DataFrame:
    df = pd.DataFrame(results, columns=columns, index=index)
    df.index.set_names(["bins"], inplace=True)
    df = df.melt(var_name=var_name, value_name="stats", ignore_index=False)
    df[var_name] = df[var_name].astype("category", copy=True)
    df.reset_index(inplace=True)
    return df


def _f_g_function(distances: NDArrayA, support: NDArrayA) -> tuple[NDArrayA, NDArrayA]:
    counts, bins = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()
    return bins, np.concatenate((np.zeros((1,), dtype=float), fracs))


def _l_function(distances: NDArrayA, support: NDArrayA, n: int, area: float) -> tuple[NDArrayA, NDArrayA]:
    n_pairs_less_than_d = (distances < support.reshape(-1, 1)).sum(axis=1)  # type: ignore[attr-defined]
    intensity = n / area
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    l_estimate = np.sqrt(k_estimate / np.pi)
    return support, l_estimate


def _l_multiple_function(distances: NDArrayA,
                         support: NDArrayA,
                         n_center: int,
                         n_neighbor: int,
                         area: float,
                         remove_diagonal: bool) -> tuple[NDArrayA, NDArrayA]:

    # break the line below less than 80  characters
    distances = distances.flatten()
    n_pairs_less_than_d = (distances < support.reshape(-1, 1)).sum(axis=1)
    if remove_diagonal:
        # Remove diagona except for when Radius = 0
        n_pairs_less_than_d = n_pairs_less_than_d - n_center
        n_pairs_less_than_d[n_pairs_less_than_d < 0] = 0
    k_estimate = n_pairs_less_than_d * area / n_center / n_neighbor
    l_estimate = np.sqrt(k_estimate / np.pi)
    return support, l_estimate


def _ppp(
        hull: ConvexHull,
        n_simulations: int,
        n_observations: int,
        rng: default_rng | None = None,
        seed: int | None = None) -> NDArrayA:
    """
    Simulate Poisson Point Process on a polygon.

    Parameters
    ----------
    hull
        Convex hull of the area of interest.
    n_simulations
        Number of simulated point processes.
    n_observations
        Number of observations to sample from each simulation.
    rng
        Random number generator, superseeds seed
    seed
        Random seed.

    Returns
    -------
    An Array with shape ``(n_simulation, n_observations, 2)``.
    """

    if rng is None:
        rng = default_rng(None if seed is None else seed)
    vxs = hull.points[hull.vertices]
    deln = Delaunay(vxs)

    bbox = np.array([*vxs.min(0), *vxs.max(0)])
    result = np.empty((n_simulations, n_observations, 2))

    for i_sim in range(n_simulations):
        i_obs = 0
        while i_obs < n_observations:
            x, y = (
                rng.uniform(bbox[0], bbox[2]),
                rng.uniform(bbox[1], bbox[3]),
            )
            if deln.find_simplex((x, y)) >= 0:
                result[i_sim, i_obs] = (x, y)
                i_obs += 1

    return result.squeeze()
