# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="AltaStata <support@altastata.com>" \
      org.opencontainers.image.title="AltaStata ChRIS Demo" \
      org.opencontainers.image.description="Sample ChRIS plugin that trains a PyTorch model against AltaStata data"

USER root

# Install Java (required for Py4J/AltaStata)
# Using OpenJDK 17 JRE headless (LTS, smaller footprint, no GUI components)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ARG SRCDIR=/usr/local/src/app
WORKDIR ${SRCDIR}

# Install dependencies as root
COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application files with proper ownership for OpenShift (UID 1001, GID 0)
COPY --chown=1001:0 . .

# Install the application package as root
ARG extras_require=none
RUN pip install --no-cache-dir ".[${extras_require}]"

# Clean up build artifacts
RUN find ${SRCDIR} -type f -name "*.pyc" -delete && \
    find ${SRCDIR} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Set proper permissions for OpenShift (UID 1001 with group 0, group read/write/execute)
RUN chown -R 1001:0 ${SRCDIR} && \
    chmod -R g+rwX ${SRCDIR} && \
    chown -R 1001:0 /usr/local/lib/python3.12/site-packages && \
    chmod -R g+rwX /usr/local/lib/python3.12/site-packages || true

# Create working directory for non-root user with OpenShift-compatible permissions
RUN mkdir -p /app && \
    chown -R 1001:0 /app && \
    chmod -R g+rwX /app

# Switch to non-root user (OpenShift uses UID 1001 with group 0)
USER 1001

# Set working directory (writable by non-root user in OpenShift)
WORKDIR /app

CMD ["altastata-chris-demo"]
