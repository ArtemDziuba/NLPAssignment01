FROM python:3.11-slim-bookworm@sha256:ef56ae98435cc40d8a27020b0b32a9941515acd5840f0fba34858bb34dd362d1

# System build dependencies live only in this image, not on the Windows host.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libboost-program-options-dev zlib1g-dev libbz2-dev liblzma-dev \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.23
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-system-dev libboost-thread-dev libboost-test-dev libboost-iostreams-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kenlm
# Build tools and Python binding from the identical locked release archive.
RUN curl -fL https://files.pythonhosted.org/packages/02/6e/f4e159cef39ea4ef2244b4aa32857d447d60a5d0ee269e1b1f8d3723c03e/kenlm-0.3.0.tar.gz -o source.tar.gz \
    && echo 'c4628bb9fb63c8a6f9240035b8b037385cfc404cb72e933cf48878291edac1e8  source.tar.gz' | sha256sum -c - \
    && tar -xzf source.tar.gz --strip-components=1 \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=3 \
    && cmake --build build --parallel 2
ENV PATH="/opt/kenlm/build/bin:${PATH}"
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock readme.md ./
RUN uv sync --locked --extra kenlm --python 3.11 --no-install-project
COPY src ./src
RUN uv sync --locked --extra kenlm --python 3.11
COPY docker/toolchain.py /opt/kenlm/record_toolchain.py
RUN /opt/venv/bin/python /opt/kenlm/record_toolchain.py
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY data/evaluation/ua_asr_hypotheses_500.jsonl ./data/evaluation/
ENV PYTHONUNBUFFERED=1
# A mounted checkout may also contain a Windows .venv: use the image environment explicitly.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["scripts/run_experiments.py", "--help"]
