import json
from pathlib import Path

import gradio as gr

with Path(__file__).with_name("examples.json").open(encoding="utf-8") as handle:
    report = json.load(handle)

examples = {row["id"]: row for row in report["examples"]}


def show(case_id):
    example = examples[case_id]
    before, after = example["before"], example["after"]
    return (
        example["document"],
        example["expected"],
        before["prediction"],
        after["prediction"],
        {
            "before": {key: before[key] for key in ("schema_valid", "exact", "unsupported")},
            "after": {key: after[key] for key in ("schema_valid", "exact", "unsupported")},
        },
    )


with gr.Blocks(title="Verifine SLM") as demo:
    gr.Markdown("# Verifine SLM")
    gr.Markdown("Compare recorded model outputs on synthetic Ukrainian and English documents.")
    first = next(iter(examples))
    select = gr.Dropdown(
        choices=[
            (f"{row['language'].upper()} · {row['task']} · {key}", key)
            for key, row in examples.items()
        ],
        value=first,
        label="Document",
    )
    document = gr.Textbox(label="Source document", lines=6, interactive=False)
    with gr.Row():
        before = gr.Code(label=report["before_model"], language="json", interactive=False)
        after = gr.Code(label=report["after_model"], language="json", interactive=False)
    with gr.Row():
        expected = gr.JSON(label="Expected output")
        checks = gr.JSON(label="Verification")
    outputs = [document, expected, before, after, checks]
    select.change(show, select, outputs)
    demo.load(show, select, outputs)

if __name__ == "__main__":
    demo.launch()
