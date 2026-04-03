"""Interactive Gradio viewer for eval datasets."""

import html
import json
from typing import Any, Optional

from datasets import load_dataset

DATASET_NAME = "predicted_spam_classification"

ROLE_COLORS = {
    "system": ("#6b7280", "#f3f4f6"),
    "user": ("#2563eb", "#eff6ff"),
    "assistant": ("#059669", "#f0fdf4"),
}
REASONING_COLOR = ("#8b5cf6", "#faf5ff")


# ---------------------------------------------------------------------------
# Parquet field helpers
# ---------------------------------------------------------------------------


def _coerce_messages(raw: Any) -> list[dict[str, str]]:
    """Extract the message list from the messages column value."""
    if isinstance(raw, dict):
        inner = raw.get("messages", [])
    else:
        inner = raw

    if isinstance(inner, str):
        inner = json.loads(inner)

    return [dict(m) for m in inner]


def _coerce_outputs(raw: Any) -> Optional[dict[str, str]]:
    """Extract the outputs dict from the messages column value."""
    if isinstance(raw, dict):
        outputs = raw.get("outputs")
        if isinstance(outputs, dict):
            return dict(outputs)
    return None


def _format_messages_html(messages_field: Any) -> str:
    msgs = _coerce_messages(messages_field)
    outputs = _coerce_outputs(messages_field)

    parts: list[str] = []
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        border, bg = ROLE_COLORS.get(role, ("#374151", "#f9fafb"))
        parts.append(
            f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
            f'background:{bg};border-radius:4px;color:#111;">'
            f'<strong style="color:{border};">{html.escape(role.upper())}</strong>'
            f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(content))}</pre>'
            f"</div>"
        )

    if outputs:
        reasoning = outputs.get("reasoning_content", "")
        text = outputs.get("text", "")
        if reasoning and str(reasoning).strip():
            border, bg = REASONING_COLOR
            parts.append(
                f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
                f'background:{bg};border-radius:4px;color:#111;">'
                f'<strong style="color:{border};">REASONING</strong>'
                f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(reasoning))}</pre>'
                f"</div>"
            )
        if text:
            border, bg = ROLE_COLORS["assistant"]
            parts.append(
                f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
                f'background:{bg};border-radius:4px;color:#111;">'
                f'<strong style="color:{border};">OUTPUT</strong>'
                f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(text))}</pre>'
                f"</div>"
            )

    return "\n".join(parts) if parts else "<p>No messages found.</p>"


def _make_banner(row_idx: int, row: Any) -> str:
    correct = row.get("correct")
    confidence = row.get("confidence")
    banner_color = "#059669" if correct else "#dc2626"
    confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
    return (
        f'<div style="padding:8px 12px;margin:0 0 8px 0;background:#f9fafb;'
        f'border-radius:4px;border:1px solid #e5e7eb;color:#111 !important;">'
        f'<strong style="color:#111;">Row {row_idx}</strong> &mdash; '
        f'<span style="color:{banner_color};font-weight:bold;">'
        f"{'Correct' if correct else 'Incorrect'}</span>"
        f' &mdash; confidence: <strong style="color:#111;">{confidence_str}</strong>'
        f"</div>"
    )


def _extract_preview(messages_field: Any, max_len: int = 120) -> str:
    outputs = _coerce_outputs(messages_field)
    text = str(outputs.get("text", "")) if outputs else ""
    if not text.strip():
        msgs = _coerce_messages(messages_field)
        text = msgs[-1].get("content", "") if msgs else ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch an interactive viewer for the eval dataset."""
    import gradio as gr
    import pandas as pd

    ds = load_dataset(DATASET_NAME)
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    # Build table rows dynamically from all columns
    display_cols = [c for c in df.columns if c != "messages"]
    headers = ["#"] + display_cols + (["preview"] if "messages" in df.columns else [])

    def _fmt_cell(val: Any) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        if isinstance(val, bool):
            return "\u2713" if val else "\u2717"
        if isinstance(val, float):
            return f"{val:.3f}"
        s = str(val)
        if len(s) > 120:
            return s[:120] + "..."
        return s

    samples: list[list[str]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        cells = [str(idx)] + [_fmt_cell(row.get(c)) for c in display_cols]
        if "messages" in df.columns:
            cells.append(_extract_preview(row.get("messages", {})))
        samples.append(cells)

    # --- Event handler ---

    def on_row_select(evt: gr.SelectData):
        row_idx = evt.index if isinstance(evt.index, int) else evt.index[0]
        row = df.iloc[row_idx]
        banner = _make_banner(row_idx, row)
        msg_html = banner + _format_messages_html(row["messages"])
        return gr.HTML(value=msg_html, visible=True)

    # --- Gradio UI ---

    with gr.Blocks(title=f"Eval Viewer \u2014 {DATASET_NAME}") as demo:
        gr.Markdown(f"# Eval Viewer \u2014 `{DATASET_NAME}`\n{len(df)} rows")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                dataset_component = gr.Dataset(
                    components=[gr.Textbox(visible=False) for _ in headers],
                    headers=headers,
                    samples=samples,
                    samples_per_page=20,
                    label="Eval Results (click a row to inspect)",
                )

            with gr.Column(scale=1):
                messages_html = gr.HTML(
                    value="<p style='color:#6b7280;'>Click a row to view messages.</p>",
                    label="Messages",
                )

        dataset_component.select(
            fn=on_row_select,
            inputs=[],
            outputs=[messages_html],
        )

    demo.launch(server_port=7860, share=False)


if __name__ == "__main__":
    main()
