# Benchmark files

`dataset-manifest.json` contains the dataset hashes used by the included runs. Regenerate the documents with `verifine data`. Each quality report includes predictions, the target values and the deterministic score breakdown. Each load report records request timings, reported token counts and sampled GPU board telemetry.

`gemma-base.json` is the 300-case unassisted reference on the installed Gemma GGUF. The earliest constrained Gemma load trials used the server's automatic reasoning budget and are retained for diagnosis. Reports generated after the launch profile was corrected include both time to the first emitted token and time to the first visible answer token.

For ablation comparisons, use reports with matching dataset hashes and decoding settings. The local electricity estimate and optional externally configured token prices use different units and different cost assumptions. [Ablation methodology](../docs/ABLATION_REPORT.md)
