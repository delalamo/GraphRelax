# GraphRelax Docker Image
# Combines RFD4-Proteina sequence design with OpenMM AMBER relaxation
#
# Build: docker build -t graphrelax .
# Run:   docker run --rm -v $(pwd):/data graphrelax -i /data/input.pdb -o /data/output.pdb

FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.source="https://github.com/delalamo/GraphRelax"
LABEL org.opencontainers.image.description="GraphRelax: RFD4-Proteina + OpenMM AMBER relaxation"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Accept version as build argument for setuptools-scm
ARG VERSION=0.0.0.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_GRAPHRELAX=${VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files 
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install conda dependencies and package
# Note: openmm and pdbfixer are only available on conda-forge, not PyPI
RUN mamba install -y -c conda-forge python=3.11 openmm pdbfixer pytorch-cpu \
    && mamba clean -afy \
    && pip install --no-cache-dir -e .

# Design/repacking requires a separately provisioned RFD4 GPU environment.
# This base image supports --no-repack without model weights.

# Create non-root user for security
RUN useradd -m graphrelax && chown -R graphrelax:graphrelax /app
USER graphrelax

ENTRYPOINT ["graphrelax"]
CMD ["--help"]
