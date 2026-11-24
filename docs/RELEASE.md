# Release artifacts

Choose a Hugging Face namespace for the adapter, quantized weights, dataset and Space. Follow the original base-model license and include it with its attribution. Use one repository per artifact type or distinct subdirectories in a model repository.

Run `python scripts/prepare-release.py --adapter artifacts/qwen-grpo --base "$VERIFINE_QWEN_MODEL"` to stage the adapter and synthetic dataset. The script verifies the base shard hashes before recording the pinned base ID and revision. It includes the base license, model card, dataset card and exact data manifest. Copy the GGUF, the applicable model card as `README.md`, and the original base license into `artifacts/release/gguf`.

After setting your own namespace and authenticating the Hugging Face CLI, use explicit artifact uploads. The model source files and local MLflow run directory can be large.

```bash
hf upload YOUR_NAME/verifine-qwen-adapter artifacts/release/adapter . --repo-type model
hf upload YOUR_NAME/verifine-qwen-gguf artifacts/release/gguf . --repo-type model
hf upload YOUR_NAME/verifine-synthetic artifacts/release/dataset . --repo-type dataset
hf upload YOUR_NAME/verifine-demo artifacts/space . --repo-type space
```

The Space uses recorded predictions from matched synthetic cases. Generate it by passing two reports and their exact dataset to `scripts/prepare-space.py`. The generated directory contains the self-contained Gradio app, its Space metadata, and synthetic examples. The Space can replay them on CPU.

Use the input dataset hash and both run manifests to reproduce the public tables. Report the context length, engine version, concurrency, sampling settings, GPU and quantization format for serving comparisons.
