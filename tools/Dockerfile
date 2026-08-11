# Dockerfile.v0.9.3 — nciccbr/spac:v0.9.3
#
# SPAC CPU image for AWS Galaxy. Adds the new CPU PhenoGraph path
# (spac.transform.clustering.phenograph.cpu) that lives on the
# feature/phenograph-gpu-refactor branch, alongside the legacy
# spac.transformations.phenograph_clustering that the existing
# phenograph_clustering.xml Galaxy tool uses.
#
# =============================================================================
# Relationship to v0.9.2
# -----------------------------------------------------------------------------
# This file is a minimal patch fork of the validated v0.9.2 Dockerfile. The only
# substantive differences are:
#
#   1. LABEL version bumped 0.9.2 -> 0.9.3
#   2. GITHUB_BRANCH remains on feature/phenograph-gpu-refactor so the image
#      picks up the PhenoGraph CPU wrapper fix for NumPy community labels.
#
# Everything else — curl-based environment.yml fetch, --no-deps pip install,
# channel config, env variables, CMD — is copied verbatim from v0.9.2 to
# minimize behavior drift and stay within the known-good CCBR/Dockers2 pattern.
#
# Why curl and not COPY?
# ----------------------
# The CCBR/Dockers2 build-docker action's build context does not reliably
# make sibling files (like environment.yml) visible to the Docker build.
# Attempting `COPY environment.yml ...` fails with
#   EnvironmentFileNotFound: '/environment.yml' file not found
# The v0.9.1 workaround — curl from raw.githubusercontent.com at build time —
# is what actually works in this CI.
# =============================================================================

FROM continuumio/miniconda3:24.3.0-0

# Build arguments
ARG ENV_NAME=spac
# Default to the feature branch until the PhenoGraph GPU refactor merges
# to SCSAWorkflow main and a release is tagged. Bump to the tag (e.g.
# v0.9.3 on SCSAWorkflow) once that happens.
ARG GITHUB_BRANCH=feature/phenograph-gpu-refactor
ARG GITHUB_REPO=FNLCR-DMAP/SCSAWorkflow

# Set labels
LABEL maintainer="FNLCR-DMAP"
LABEL description="SPAC - Single Cell Spatial Analysis Container"
LABEL version="0.9.3"
LABEL github.branch="${GITHUB_BRANCH}"

# Install system dependencies including Chromium for Kaleido (ARM64 compatible)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium for Kaleido visualization support (ARM64 compatible)
RUN apt-get update && \
    apt-get install -y chromium && \
    rm -rf /var/lib/apt/lists/*

# Set Chrome binary path for Kaleido
ENV CHROME_BIN=/usr/bin/chromium

# Install and configure libmamba solver
RUN conda install -n base -y conda-libmamba-solver && \
    conda config --set solver libmamba

WORKDIR /opt/SCSAWorkflow

# Step 1: Fetch environment.yml from GitHub branch
# (Replaces local COPY since we're installing from GitHub, and because the
#  CCBR/Dockers2 build context does not include sibling files.)
RUN curl -fsSL "https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/environment.yml" \
    -o environment.yml

# Step 2: Follow README.md instructions exactly
# "If conda is not activate" - conda activate (already active in base)

# Step 3: "Adding constumized scimap conda pacakge channel supported by DMAP"
RUN conda config --add channels https://fnlcr-dmap.github.io/scimap/ && \
    conda config --add channels conda-forge && \
    conda config --add channels ohsu-comp-bio && \
    conda config --add channels leej3 && \
    conda config --add channels bioconda

# Step 4: "Create the Conda environment from environment.yml"
# Set SSL verification to false for problematic channels
RUN conda config --set ssl_verify false && \
    conda env create -f environment.yml && \
    conda clean -afy && \
    conda config --set ssl_verify true

# Step 5: Make the environment available (simulate "conda activate spac")
# Set all necessary environment variables for conda activation
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV CONDA_PREFIX=/opt/conda/envs/${ENV_NAME}
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}
ENV PYTHONPATH=/opt/conda/envs/${ENV_NAME}/lib/python3.9/site-packages:${PYTHONPATH}

# Step 6: "Install the SPAC package" from GitHub branch
# (Replaces: pip install -e . with direct GitHub install)
# Using --no-deps to preserve conda-installed package versions
# (pandas 1.5.3, scimap 2.1.3_dmap_pandas153) that would otherwise be
# clobbered by spac's unpinned install_requires.
RUN /opt/conda/envs/${ENV_NAME}/bin/pip install --no-deps \
    "git+https://github.com/${GITHUB_REPO}.git@${GITHUB_BRANCH}"

# Make conda activate spac environment automatically in bash shells
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc

# Create a wrapper script that ensures conda environment is activated
# This helps when Galaxy runs commands directly
RUN echo '#!/bin/bash' > /usr/local/bin/conda-run && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /usr/local/bin/conda-run && \
    echo 'conda activate spac' >> /usr/local/bin/conda-run && \
    echo 'exec "$@"' >> /usr/local/bin/conda-run && \
    chmod +x /usr/local/bin/conda-run

# Set the shell to use bash for RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables for headless notebook execution
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

# Create working directories for volume mapping
RUN mkdir -p /workspace /data /results

# Install jupyter and nbconvert for notebook testing
RUN /opt/conda/envs/${ENV_NAME}/bin/pip install jupyter nbconvert

# -----------------------------------------------------------------------------
# Verify SPAC installation works correctly in multiple ways.
#
# Extended from v0.9.1 with a check for the new CPU path:
#   - Test 5: new CPU path (spac.transform.clustering.phenograph.cpu +
#             phenograph_clustering_cpu_template) — only on the feature
#             branch. If Test 5 fails with ModuleNotFoundError on
#             `spac.transform`, the GITHUB_BRANCH build arg is pointing
#             at a ref that predates the PhenoGraph GPU refactor.
# -----------------------------------------------------------------------------
RUN echo "=== VERIFYING SPAC INSTALLATION ===" && \
    echo "Test 1: Direct python call" && \
    python -c "import spac; print(f'SPAC version: {spac.__version__}')" && \
    echo "Test 2: Which python" && \
    which python && \
    echo "Test 3: Python path" && \
    python -c "import sys; print(sys.executable)" && \
    echo "Test 4: Import scimap" && \
    python -c "import scimap; print('scimap imported successfully')" && \
    echo "Test 5: New CPU path (transform/clustering/phenograph)" && \
    python -c "from spac.transform.clustering.phenograph import phenograph_cpu, prepare_features; \
               from spac.templates.phenograph_clustering_cpu_template import run_from_json; \
               print('new CPU path OK')" && \
    echo "Test 6: Verify UMAP compatibility with scikit-learn" && \
    python -c "import umap; import numpy as np; data = np.random.rand(100, 10); reducer = umap.UMAP(n_neighbors=5, min_dist=0.1); reducer.fit_transform(data); print(f'UMAP version: {umap.__version__} - OK')" && \
    echo "=== ALL TESTS PASSED ==="

# Set working directory for Jupyter (will be mounted via volume)
WORKDIR /workspace

# Default command starts Jupyter notebook server
CMD ["/opt/conda/envs/spac/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
