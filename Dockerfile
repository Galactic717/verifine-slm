FROM python:3.12-slim-trixie
# Install the Debian security update for CVE-2026-124851.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade libzstd1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir . && useradd --create-home verifine
USER verifine
EXPOSE 8088
CMD ["verifine", "gateway", "--host", "0.0.0.0"]
