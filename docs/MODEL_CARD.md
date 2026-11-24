---
language:
  - uk
  - en
pipeline_tag: text-generation
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - qlora
  - grpo
  - structured-extraction
---

# Verifine SLM model card

The intended adapter attaches to Qwen3‑4B‑Instruct‑2507. The experiments also include a standalone Gemma 4 E4B GGUF baseline. Original model licenses and attribution travel with distributed derivatives. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [Gemma model card](https://huggingface.co/google/gemma-4-E4B-it)

## Intended use

Extract facts from Ukrainian and English invoices, fictional clinical notes and agreements, and propose ticket tool arguments. An application must independently authorize every real action. Clinical examples test transcription of fictional facts. Diagnosis, treatment selection and use of actual patient data are outside the evaluation scope.

## Training and evaluation data

The seeded factory renders four task families with explicit distractors, absent fields and different document layouts for each split. It uses synthetic entities and document IDs. Labels are checked against the same strict schema and deterministic text-evidence rules used by the reward. The held-out set has 300 cases. Template families and near-duplicate checks reduce overlap. Shared fact types, cue words and generation logic limit external validity.

## Verification and failure modes

The schema checks types and keys. The grounding heuristic checks source spans, aliases and normalized amounts. It can miss negation, value attribution, spelling variants and OCR errors. A value may appear in the document under the wrong role. The labeled reward addresses that distinction on synthetic cases. Required nullable fields allow honest abstention. Invalid structured predictions count as errors in quality evaluation.

Use the evaluation harness and gateway with documents representative of the intended workflow. Gateway metrics record durations, statuses and backend token counts. The local setup binds to loopback and the remote evaluation path accepts only synthetic cases. [Measured results](ABLATION_REPORT.md)
