---
language:
  - uk
  - en
task_categories:
  - text-generation
tags:
  - synthetic
  - bilingual
  - structured-extraction
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: test
        path: golden.jsonl
---

# Verifine bilingual extraction data

The dataset has 787 training, 118 validation and 300 golden examples at seed 42. The test split includes 150 Ukrainian and 150 English documents. Each row has an ID, template group, task, language, synthetic marker, text and expected structured fields. The four tasks cover invoices, fictional clinical notes, contracts and support tool proposals.

The generator uses invented organizations and case identifiers. It produces text with optional missing facts, formatting variations and distractor values. A grouped template assignment and a shingle similarity audit reduce near-duplicate leakage between splits. Train and test share the generator and field cues. Performance on this dataset has limited predictive value for authentic operational documents.

The expected JSON is validated with Pydantic and deterministic text evidence rules. The model's structured output is scored against those labels. Clinical notes are fictional extraction examples. Support tool fields are proposals for an external approval workflow. See the manifest for the exact file hashes of this release.
