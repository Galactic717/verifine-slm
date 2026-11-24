FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir . && useradd --create-home verifine
USER verifine
EXPOSE 8088
CMD ["verifine", "gateway", "--host", "0.0.0.0"]
