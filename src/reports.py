from __future__ import annotations

from typing import Dict, List


def build_report_markdown(timestamp: str, model_path: str, pred: Dict, class_names: List[str], sequence_length: int) -> str:
    label = pred.get("label", "N/A")
    probs = pred.get("probs", {})
    thr = pred.get("threshold", "N/A")

    md = f"""
# Driver Behavior Detection Report (Placeholder)

**Generated:** {timestamp}  
**Model:** `{model_path}`  
**Sequence length:** {sequence_length}  
**Classes:** {", ".join(class_names)}

---

## Prediction
- **Predicted label:** **{label}**
- **Probabilities:** {probs}
- **Decision threshold:** {thr}

---

## Notes
This is a **web report placeholder**. In Topic 8, it can be extend to PDF/DOCX and embed images.
"""
    return md.strip() + "\n"


def save_text_report(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
