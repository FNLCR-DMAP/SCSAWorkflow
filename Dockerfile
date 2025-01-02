# Use the ubuntu 20.04 system
FROM ubuntu:20.04

# Set system overall permission in the image
RUN chmod -R ugo+rx /root \
    && chmod -R ugo+rx /opt \
    && chmod -R ugo+rx /tmp \
    && umask u=rwx,g=rwx,o=rx \
    && echo "TMPDIR=/mnt" > /root/.Renviron

ARG DEBIAN_FRONTEND=noninteractive

ARG CONDA_VERSION_INTSALL="Miniconda3-py38_23.9.0-0-Linux-x86_64.sh"

RUN apt-get update  && \
    apt-get install -y --no-install-recommends \
      ed \
      less \
      locales \
      vim-tiny \
      wget \
      ca-certificates \
      fonts-texgyre libx11-dev && \
      rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      dirmngr \
      curl \
      libcurl4-openssl-dev && \
    add-apt-repository --enable-source --yes "ppa:marutter/rrutter4.0" && \
    add-apt-repository --enable-source --yes "ppa:c2d4u.team/c2d4u4.0+"
    
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
  locale-gen en_US.utf8 && \
	/usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libx11-6 \
      libxss1 \
      libxt6 \
      libxext6 \
      libsm6 \
      libice6 \
      xdg-utils \
      libxt-dev \
      xorg-dev \
      libcairo2 \
      libcairo2-dev \
      libpango1.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \ 
    apt install -y --no-install-recommends \
      zlib1g-dev \
      libcurl4-openssl-dev \
      libxml2-dev \
      libssl-dev \
      libpng-dev \
      libhdf5-dev \
	    libquadmath0 \
	    libtiff5-dev \
	    libjpeg-dev \
	    libfreetype6-dev \
	    libgfortran5 \
	    libgmp-dev \
	    libmpc-dev \
	    libopenblas0-pthread \
	    libgeos-dev \
      cmake \
      libfftw3-dev \
      git

RUN wget https://repo.anaconda.com/miniconda/$CONDA_VERSION_INTSALL \
    && bash $CONDA_VERSION_INTSALL -b -p "/conda"\
    && rm -f $CONDA_VERSION_INTSALL
    
# Add miniconda to PATH and install miniconda
ENV PATH="/conda/bin:${PATH}"

# Add bioconda and conda-forge to Conda channels and install mamba
RUN conda config --append channels bioconda \
    && conda config --append channels conda-forge \
    && conda config --append channels leej3 \
    && conda install -n base -y conda-libmamba-solver\
    && conda config --set solver libmamba \
    && conda install -c conda-forge mamba conda-build
    
RUN mkdir -p /local_channel/linux-64
COPY *.tar.bz2 /local_channel/linux-64
RUN conda index --verbose /local_channel \
    && conda config --add channels file:///local_channel \
    && chmod -R ugo+rx /local_channel

# Initiate Conda bash terminal and create conda environment using environment.yml file
COPY environment.yml .

RUN conda env create -f environment.yml --prefix /conda/envs/SCSAWorkflow_NIDAP \
    && conda install -p /conda/envs/SCSAWorkflow_NIDAP -c conda-forge python-semantic-release \
    && conda install -p /conda/envs/SCSAWorkflow_NIDAP -c conda-forge flake8 \
    && conda clean --all \
    && conda config --add channels https://fnlcr-dmap.github.io/scimap/


# Ask conda to activate the built environment when starting
RUN ENV_NAME=$(grep -E '^name:' environment.yml | sed 's/name: *//') \
    && echo "source activate $ENV_NAME" > ~/.bashrc
    
# Create symbolic link to simulate if conda is installed into root
RUN mkdir -p /root/miniconda3/etc/profile.d \
    && ln -s /conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh \
    && mkdir -p /conda/conda-bld \
    && mkdir -p /root/miniconda3/conda-bld \
    && ln -s /conda/conda-bld /root/miniconda3/conda-bld

ENV PATH="/conda/envs/SCSAWorkflow_NIDAP/bin:$PATH"

RUN chmod -R ugo+rwx /conda

CMD ["bash"]