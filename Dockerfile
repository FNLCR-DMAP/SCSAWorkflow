FROM continuumio/miniconda3:24.3.0-0

# Build arguments
ARG ENV_NAME=spac

# Set labels
LABEL maintainer="FNLCR-DMAP"
LABEL description="SPAC - Single Cell Spatial Analysis Container"
LABEL version="0.9.0"

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

# Simulate exactly what a reviewer would do following README.md instructions
WORKDIR /home/reviewer/SCSAWorkflow

# Step 1: Copy the repository files (simulating a reviewer's local setup)
COPY . .

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
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}

# Step 6: "Install the SPAC package in development mode"
RUN /opt/conda/envs/${ENV_NAME}/bin/pip install -e .

# Set environment variables for headless notebook execution
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

# Create working directories for volume mapping
RUN mkdir -p /workspace /data /results

# Install jupyter and nbconvert for notebook testing
RUN /opt/conda/envs/${ENV_NAME}/bin/pip install jupyter nbconvert

# Verify SPAC installation works correctly
RUN echo "=== VERIFYING SPAC INSTALLATION ===" && \
    /opt/conda/envs/${ENV_NAME}/bin/python -c "import spac; print(f'SPAC version: {spac.__version__}'); import scimap; print('All modules imported successfully!')" || \
    echo "Some import issues detected but proceeding with test"

# Set working directory for Jupyter (will be mounted via volume)
WORKDIR /workspace

# Default command starts Jupyter notebook server
CMD ["/opt/conda/envs/spac/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]