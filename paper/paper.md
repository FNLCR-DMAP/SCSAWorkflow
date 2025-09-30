---
title: 'SPAC: A Python Package for Spatial Single-Cell Analysis of Multiplexed Imaging'
tags:
  - multiplexed imaging
  - spatial proteomics
  - single-cell analysis
  - tumor microenvironment
authors:
  - name: Fang Liu
    orcid: 0000-0002-4283-8325
    affiliation: 1
  - name: Rui He
    affiliation: 2
  - name: Andrei Bombin
    affiliation: 3
  - name: Ahmad B. Abdallah
    affiliation: 4
  - name: Omar Eldaghar
    affiliation: 4
  - name: Tommy R. Sheeley
    affiliation: 4
  - name: Sam E. Ying
    affiliation: 4
  - name: George Zaki
    orcid: 0000-0002-2740-3307
    corresponding: true
    affiliation: 1
affiliations:
  - index: 1
    name: Frederick National Laboratory for Cancer Research, United States
  - index: 2
    name: Essential Software Inc., United States
  - index: 3
    name: Axle Informatics, United States
  - index: 4
    name: Purdue University, United States
date: 15 April 2025
bibliography: paper.bib
---

# Summary

Multiplexed immunofluorescence microscopy captures detailed measurements of spatially resolved, multiple biomarkers simultaneously, revealing tissue composition and cellular interactions in situ among single cells. The growing scale and dimensional complexity of these datasets demand reproducible, comprehensive and user-friendly computational tools. To address this need, we developed SPAC **(SPA**tial single-**C**ell analysis), a Python-based package and a corresponding Shiny application within an integrated, modular SPAC ecosystem designed specifically for biologists without extensive coding expertise. Following image segmentation and extraction of spatially resolved single-cell data, SPAC streamlines downstream phenotyping and spatial analysis, facilitating characterization of cellular heterogeneity and spatial organization within tissues. Through scalable performance, specialized spatial statistics, highly customizable visualizations, and seamless workflows from dataset to insights, SPAC significantly lowers barriers to sophisticated spatial analyses.

# Statement of Need

Multiplexed protein imaging (e.g., CODEX [@Goltsev:2018], CyCIF [@Lin:2018], MxIF [@Gerdes:2013]) yields gigabyte-scale, single-cell datasets with millions of cells per study. Extracting biological signal from these data requires reproducible workflows that span preprocessing, phenotyping, spatial statistics, and interactive visualization. While recent tools have advanced the analysis of spatial omics, biologists without coding expertise still face steep barriers to conduct end-to-end analyses independently and iteratively. This slows hypothesis testing and risks brittle, one-off scripts that are hard to reuse. SPAC is designed to close this gap.

SPAC is a Python package for downstream analysis of single-cell multiplexed imaging after segmentation. It ingests per-cell tables (CSV/H5AD/tabular) with spatial coordinates and marker intensities and exposes a reproducible, AnnData-compliant Python API. All analysis artifacts and figures are preserved in AnnData, supporting sharing, re-runs, and downstream reuse. As the core analytical layer of the broader SPAC ecosystem [@Liu:2025], the package connects to interactive pipelines and real-time dashboards on enterprise platforms via a modular, layered architecture, enabling large-scale analyses without requiring users to write code. Its purpose is to lower the barrier for non-programmers while preserving rigorous, reusable workflows; for example, SPAC uses biologist-friendly terminology (e.g., "cells," "features," "tables," "associated tables," "annotations") and clear messages so users can interpret results and diagnostics without a bioinformatic analyst's assistance.

To address real-time scalability challenges in analyzing large multiplex imaging datasets (exceeding 10 million cells), SPAC integrates optimized numerical routines from NumPy's compiled C-based backend, making common plotting and summarization (e.g., histograms, box plots) substantially faster, over 5× quicker than comparable Seaborn-based workflows in our tests, reducing typical runtimes from tens of seconds to a few seconds. This responsiveness supports iterative exploratory data analysis and figure generation on very large cohorts without sacrificing reproducibility.

SPAC provides customizable visualization methods. Pin colors to annotations maintain consistent color mapping across figures and sessions. Interactive spatial plots (Plotly) allow users to toggle off features (e.g., biomarkers) and switch among multiple annotations simultaneously, enabling intuitive exploration of spatial relationships and patterns. Stratified and gridded plotting with optimized statistical summary make subgroup comparisons (e.g., across conditions or phenotypes) clear within a single view.

# State of the Field

The spatial omics ecosystem includes general frameworks for single cell data and spatial analysis (AnnData/Scanpy, Squidpy [@Palla:2022]) and toolkits focused on multiplex imaging and spatial proteomics (SCIMAP [@Nirmal:2024]), alongside methods aimed primarily at spatial transcriptomics (Seurat [@Hao:2021], GraphST [@Long:2023], Bento [@Mah:2024]). These packages collectively cover data structures, clustering, dimensionality reduction, neighborhood graphs, and spatial interaction metrics, and they are widely adopted for programmatic analyses.

Within this landscape, SPAC builds directly on community standards such as AnnData and established methods, but contributes a coherent foundation that enhances key functions for spatial analyses and visualization. Highlighting a few examples, SPAC enhances core analyses (e.g., nearest-neighbor metrics leveraging SCIMAP's spatial-distance utilities) with flexible, built-in plotting for publication ready figures, including native support for stratification, subsetting, and faceting. Users can fine-tune plot aesthetics and export layout (margins, size, DPI) to produce clear, consistent figures. SPAC strengthens the functionality of existing packages: SPAC implements a phenotype-pair specific Ripley's L that treats one phenotype as "centers" and another as "neighbors," and applies radius-dependent guard-zone edge correction by excluding center cells whose r-neighborhood intersects the ROI boundary. Unlike generalized implementations (e.g., Squidpy), this design reduces edge-induced inflation at larger radii and yields more reliable, interpretable L(r) curves for concrete cell-cell interactions. SPAC adds new features. Among its features, SPAC implements efficient KDTree-based neighborhood profiling that quantifies the distribution of neighboring cell phenotypes in user-defined radial distance bins around each center cell of a specified phenotype. The resulting three-dimensional array, which summarizes the local cellular microenvironment, is stored in the AnnData object and supports dimensionality-reduction methods such as spatial UMAP [@Giraldo:2021].

# Structure and Implementation

The SPAC package is available at [GitHub](https://github.com/FNLCR-DMAP/SCSAWorkflow) and can be installed locally via conda. It comprises five modules that streamline data processing, transformation, spatial analysis, visualization, and common utilities. All modules are interoperable, forming a cohesive workflow \autoref{fig:workflow}.

At the architectural level, the SPAC Python package serves as a foundation layer that builds on community libraries while underpinning interactive, no-code surfaces. By decoupling the analysis layer from any specific UI, a single, tested codebase can power both web-hosted pipelines and interactive dashboards, eliminating duplicated analysis logic. This separation keeps the package focused on analysis semantics and performance, while allowing heterogeneous front ends to reuse the same analysis primitives.

SPAC adheres to enterprise-level software engineering standards, featuring extensive unit testing, rigorous edge-case evaluation, comprehensive logging, and clear, context-rich error handling. These practices ensure reliability, adaptability, and ease-of-use across various deployment environments, including interactive Jupyter notebooks, analytic platforms (e.g., Code Ocean [@CodeOcean], Palantir Foundry [@PalantirTechnologies]), Galaxy [@Jalili:2020], and real-time dashboards such as Shiny. Emphasizing readability and maintainability, SPAC provides a versatile and enhanced analytical solution for spatial single-cell analyses. To date, SPAC has been used in the analysis of over 8 datasets with over 30 million cells across diverse studies [@Keretsu:2025].

![Overview of the SPAC Workflow. The schematic presents an integrated pipeline for spatial single-cell analysis. Segmented cell data with spatial coordinates from various imaging platforms are ingested, normalized, clustered and phenotyped, and analyzed spatially to assess cell distribution and interactions while maintaining consistent data lineage.\label{fig:workflow}](figure.tif)

# Acknowledgements

We thank our collaborators at the National Cancer Institute Frederick National Laboratory, the Purdue Data Mine program, and the single-cell and spatial imaging communities for their essential contributions and resources.

# References
