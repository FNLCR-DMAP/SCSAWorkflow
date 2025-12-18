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
    orcid: 0000-0002-2610-1411
    affiliation: 2
  - name: Andrei Bombin
    orcid: 0000-0003-2445-304X
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

Multiplexed immunofluorescence microscopy captures detailed, spatially resolved measurements of multiple biomarkers simultaneously. These measurements reveal tissue composition and cellular interactions in situ at the single-cell level. The growing scale and dimensional complexity of these datasets demand reproducible, comprehensive, and user-friendly computational tools. To address this need, we developed SPAC (**SPA**tial single-**C**ell analysis), a Python-based package and a corresponding Shiny application within an integrated, modular SPAC ecosystem designed specifically for biologists without extensive coding expertise. Following image segmentation and extraction of spatially resolved single-cell data, SPAC streamlines downstream phenotyping and spatial analysis, facilitating the characterization of cellular heterogeneity and spatial organization within tissues. Through scalable performance, specialized spatial statistics, highly customizable visualizations, and seamless workflows from dataset to insights, SPAC significantly lowers the barriers to sophisticated spatial analyses.

# Statement of Need

Multiplexed protein imaging (e.g., CODEX [@Goltsev:2018], CyCIF [@Lin:2018], MxIF [@Gerdes:2013]) yields gigabyte-scale, single-cell datasets with millions of cells per study. Extracting biological signal from these datasets requires reproducible workflows that span preprocessing, phenotyping, spatial statistics, and interactive visualization.

The spatial omics ecosystem includes general frameworks for single-cell data and spatial analysis (AnnData [@Virshup:2024], Scanpy [@Wolf:2018], Squidpy [@Palla:2022]) and toolkits focused on multiplex imaging and spatial proteomics (SCIMAP [@Nirmal:2024]), alongside methods aimed primarily at spatial transcriptomics (Seurat [@Hao:2021], Giotto Suite [@Chen:2025], GraphST [@Long:2023], Bento [@Mah:2024]). These packages collectively provide data structures, clustering, dimensionality reduction, neighborhood graphs, and spatial interaction metrics, and they are widely adopted for programmatic analyses.

Despite this progress, biologists without coding expertise face steep barriers when conducting end-to-end analyses independently and iteratively. Projects often require stitching together multiple libraries and ad hoc scripts, which slows hypothesis testing, complicates reproducibility, and makes figure generation brittle and hard to reuse at scale.

SPAC is a Python package for downstream analysis of single-cell multiplexed imaging data following segmentation. It ingests per-cell tables (CSV/H5AD/tabular) with spatial coordinates and marker intensities and exposes a reproducible, AnnData‑compliant Python API; all analysis artifacts and figures are preserved to support sharing, reruns, and downstream reuse. As the core analytical layer of the broader SPAC ecosystem [@Liu:2025], the package connects to interactive pipelines and real‑time dashboards on enterprise platforms through a modular, layered architecture. This design enables large‑scale analyses without requiring users to write code. Its purpose is to lower the barrier for non‑programmers while preserving rigorous, reusable workflows. For example, SPAC uses biologist-friendly terminology (e.g., "cells," "features," "tables," "associated tables," and "annotations") and clear messaging so users can interpret results and diagnostics without a bioinformatics analyst's assistance.

To address real-time scalability for datasets exceeding 10 million cells, SPAC integrates optimized numerical routines from NumPy's compiled, C-based backend. These optimizations make common plotting and summarization tasks (e.g., histograms, box plots) substantially faster—more than 5× faster than comparable Seaborn‑based workflows in our tests—reducing typical runtimes from tens of seconds to a few seconds. This responsiveness supports iterative exploratory data analysis and figure generation on very large cohorts without sacrificing reproducibility.

SPAC provides customizable visualization methods. Pinning colors to annotations maintains consistent color mapping across figures and sessions. Interactive spatial plots (Plotly) allow users to toggle features (e.g., biomarkers) on and off and to switch among multiple annotations simultaneously, enabling intuitive exploration of spatial relationships and patterns. Stratified and gridded plotting with optimized statistical summaries makes subgroup comparisons (e.g., across conditions or phenotypes) clear within a single view.

SPAC builds directly on community standards and methods but contributes a coherent foundation that enhances key functions for spatial analyses and visualization. To highlight a few examples, SPAC enhances core analyses (e.g., nearest‑neighbor metrics leveraging SCIMAP’s spatial‑distance utilities) with flexible, built‑in plotting and native support for stratification, subsetting, and faceting. Users can fine-tune plot aesthetics and export layouts, making it straightforward to produce consistent, publication‑ready figures. SPAC strengthens the functionality of existing packages: SPAC implements a phenotype‑pair‑specific Ripley’s L that treats one phenotype as “centers” and another as “neighbors,” and applies radius‑dependent guard-zone edge correction by excluding center cells whose r‑neighborhood intersects the ROI boundary. Unlike generalized implementations (e.g., Squidpy), this design reduces edge‑induced inflation at larger radii and yields more reliable, interpretable L(r) curves for concrete cell-cell interactions. SPAC also adds new features; for instance, it implements a neighborhood profile via a KDTree‑based approach, quantifying the distribution of neighboring cell phenotypes within user-defined distance bins. The resulting three-dimensional array encodes, for every cell and every distance bin, the count of neighboring cells of each phenotype. This array captures the local cellular microenvironment within the AnnData object and supports downstream dimensionality-reduction methods such as spatial UMAP [@Giraldo:2021].

# Structure and Implementation

The SPAC package is available on [GitHub](https://github.com/FNLCR-DMAP/SCSAWorkflow) and can be installed locally via Conda or Docker images. It comprises five modules that streamline data processing, transformation, spatial analysis, visualization, and common utilities. All modules are interoperable, forming a cohesive workflow (\autoref{fig:workflow}). 

At the architectural level, the SPAC package serves as a foundation layer that builds on community libraries while underpinning interactive, no‑code surfaces. By decoupling the analysis layer from any specific UI, a single, tested codebase can power both web‑hosted pipelines and interactive dashboards, eliminating duplicated analysis logic. This separation keeps the package focused on analytical semantics and performance while allowing heterogeneous front ends to reuse the same analysis primitives. For instance, an interactive Shiny for Python dashboard ([GitHub](https://github.com/FNLCR-DMAP/SPAC_Shiny)) is hosted on Posit Connect and is accessible via a web browser at [SPAC Interactive Visualization](https://appshare.cancer.gov/spac-interactive-visualization/) [@Liu:2025]. This dashboard provides a no-code interface to SPAC's real-time visualization capabilities, featuring multiple tool modules and "Getting Started" tutorials that are publicly available to researchers without programming expertise.

SPAC adheres to enterprise-level software engineering standards, featuring extensive unit testing, rigorous edge-case evaluation, comprehensive logging, and clear, context-rich error handling. These practices ensure reliability, adaptability, and ease of use across various deployment environments, including interactive Jupyter notebooks, analytic platforms (e.g., Code Ocean [@CodeOcean], Palantir Foundry [@PalantirTechnologies]), Galaxy [@Jalili:2020], and real-time dashboards. Emphasizing readability and maintainability, SPAC provides a versatile and enhanced analytical solution for spatial single-cell analyses. To date, SPAC has been used in the analysis of more than eight datasets comprising over 30 million cells across diverse studies [@Keretsu:2025].

![Overview of the SPAC workflow. The schematic presents an integrated pipeline for spatial single-cell analysis. Segmented cell data with spatial coordinates from various imaging platforms are ingested, normalized, clustered, phenotyped, and analyzed spatially to assess cell distributions and interactions while maintaining consistent data lineage.\label{fig:workflow}](figure.png)

# Acknowledgements

We thank our collaborators at the National Cancer Institute, Frederick National Laboratory, the Purdue Data Mine program, and the single-cell and spatial imaging communities for their essential contributions and resources.

# References
