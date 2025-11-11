# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="AltaStata <support@altastata.com>" \
      org.opencontainers.image.title="AltaStata ChRIS Demo" \
      org.opencontainers.image.description="Sample ChRIS plugin that trains a PyTorch model against AltaStata data"

ARG SRCDIR=/usr/local/src/app
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

CMD ["altastata-chris-demo"]
