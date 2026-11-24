# Ablation report

## Protocol

- Hardware is a GeForce RTX 5080 with 16 GB VRAM. The Gemma run used Windows and llama.cpp build 12242. Qwen training uses WSL2, CUDA 12.8 wheels, Transformers 5.5.4 and TRL 0.29.0.
- Seed 42 generated 787 training, 118 validation and 300 golden documents after deduplication. The golden split has 150 examples per language. It exercises invoices, fictional clinical notes, contracts and support tool arguments.
- Quality reports use the entire golden split, a 240-token completion cap and greedy decoding. JSON validity includes any strict JSON value. Schema validity additionally enforces the task's Pydantic model. Macro-F1 averages task-field F1 and gives a wrong populated value one false positive and one false negative. Correct nulls enter abstention recall and exact document accuracy.
- The unsupported-value rate divides unsupported asserted values by asserted values from schema-valid predictions. Schema failures reduce F1 and exact accuracy. A wrong value that appears elsewhere in the source can still pass the grounding heuristic.
- Serving profiles report streaming completions, usable structured throughput, board memory and an electricity-only estimate based on sampled board power. The selected comparison tariff is USD 0.15/kWh. System power, amortization, hosting and network costs are outside that estimate. Model costs and clinical utility require measurements on a representative workload.

## Quality on 300 golden examples

| Version | JSON valid | Schema valid | Field macro-F1 | Exact document | Unsupported values |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E4B UD-Q8_K_XL, unassisted | 100% | 91.33% | 0.9553 | 74.00% | 0.97% |
| Qwen3-4B base NF4 | 100% | 98.67% | 0.9468 | 77.67% | 2.44% |
| Qwen3-4B SFT NF4 | 100% | 100% | 0.9839 | 92.67% | 0.10% |
| Qwen3-4B SFT + GRPO NF4 | 100% | 100% | 0.9853 | 93.00% | 0.10% |

Ukrainian field F1 was 0.9600 and English field F1 was 0.9487. Support calls had 67.57% schema validity and accounted for 24 of 26 invalid objects. The model frequently emitted a null `arguments` object or a string `"null"` as the tool name. The typed contract requires an arguments object with nullable inner fields.

The first guided-decoding benchmark used the installed build without an explicit reasoning token budget. It reached the generation cap in 44 of 48 attempts at concurrency four. The launcher now sets the reasoning budget to zero. The raw trial remains available with its original settings as a diagnostic.

SFT improved exact document accuracy by 15.00 percentage points on Qwen. The paired bootstrap 95% interval was 10.67–19.67 points and the exact McNemar p-value was 2.14 × 10⁻¹⁰. GRPO improved exact document accuracy by 0.33 points on one changed document, with McNemar p = 1.0. Its macro-F1 change was 0.0014, with a 95% paired bootstrap interval from 0.0000 to 0.0037. These results describe a seeded synthetic workload.

## Training runs

| Run | Optimizer steps | Steps with nonzero gradients | Peak CUDA reservation |
| --- | ---: | ---: | ---: |
| QLoRA SFT | 64 | 64 | 6,516 MiB |
| GRPO, original training cases | 20 | 0 | 4,894 MiB |
| GRPO, archival distractors on training cases | 20 | 2 | 4,964 MiB |

The original GRPO run generated 80 completions, all with reward 1. The saved adapter matched the SFT adapter byte-for-byte. Adding archival material produced reward variance in two of 20 groups and changed the adapter SHA-256. That checkpoint is used in the final Qwen ablation and the exported GGUF. [Training records](../benchmarks/training-runs.json)

## Serving on the same Qwen Q4_K_M GGUF

Both backends served the merged SFT + GRPO checkpoint with JSON-schema constraints, four concurrent requests, eight warmups and 48 measured requests. Request rate counts completions, and valid request rate also requires the full Pydantic contract.

| Engine | Requests/s | Schema-valid requests/s | TTFT p95 | TPOT p50 | End-to-end p95 | Peak board VRAM | GPU electricity per million requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp build 12242 | 7.33 | 7.18 | 177 ms | 11.35 ms | 779 ms | 13,389 MiB | $0.78 |
| Ollama 0.40.18 | 4.21 | 4.12 | 3,208 ms | 13.86 ms | 3,659 ms | 13,387 MiB | $0.80 |

The bench captures board-wide VRAM from `nvidia-smi`, including other GPU allocations at sampling time. The latency figures describe this Windows host and Docker Desktop configuration. The local engine profiles exceeded the brief's target end-to-end p95 of 450 ms on this workload. [llama.cpp run](../benchmarks/qwen-q4-llama-c4.json) · [Ollama run](../benchmarks/qwen-q4-ollama-c4.json)

Separate quality runs evaluated all 300 cases on the exported GGUF. Both servers produced 99.67% schema-valid answers. llama.cpp reached 0.9796 field macro-F1 and 92.00% exact document accuracy. Ollama reached 0.9735 F1 and 91.00% exact document accuracy. The local electricity estimates were $0.00177 and $0.00183 per million total tokens, respectively. [Cost and quality plot](../benchmarks/plots/serving/cost-vs-f1.png)

The FastAPI extraction route and the existing Verstak HTTP router were smoke-tested against the exported model. The Verstak request returned a valid invoice proposal and the correct currency and amount. It left the requested invoice reference empty on that fixture.

The vLLM and SGLang Compose configurations validated. Their multi-gigabyte image downloads did not finish during the experiment, so the table contains measured llama.cpp and Ollama results. The cloud baseline requires provider credentials and an explicit remote run. AWQ and FP8 have export commands. The verified post-training export here is merged Safetensors and Q4_K_M GGUF.

## Reproducibility

The dataset manifest and raw quality and serving responses include the input SHA-256. Use the report fields `backend`, `constrained`, `concurrency` and `max_tokens` before comparing latencies. The Qwen paired ablation compares before and after models with the same quantization and tokenizer. Different models and formats in the serving table must be reported with their quantization.
