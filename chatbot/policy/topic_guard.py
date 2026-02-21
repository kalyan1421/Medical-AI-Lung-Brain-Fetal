from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ALLOWED_TOPICS: Tuple[str, ...] = (
    "brain_tumor",
    "pneumonia",
    "fetal_ultrasound",
)

TOPIC_DISPLAY_NAMES: Dict[str, str] = {
    "brain_tumor": "Brain Tumor",
    "pneumonia": "Pneumonia",
    "fetal_ultrasound": "Fetal Ultrasound",
}

TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "brain_tumor": (
        "brain",
        "tumor",
        "glioma",
        "meningioma",
        "pituitary",
        "mri",
        "intracranial",
        "no tumor",
    ),
    "pneumonia": (
        "pneumonia",
        "lung",
        "lungs",
        "chest",
        "xray",
        "x-ray",
        "respiratory",
        "normal lung",
    ),
    "fetal_ultrasound": (
        "fetal",
        "foetal",
        "ultrasound",
        "pregnancy",
        "head contour",
        "segmentation",
        "sonography",
    ),
}


@dataclass
class TopicDecision:
    topic: str
    in_scope: bool
    redirected: bool
    reason: str


class TopicGuard:
    """Enforces three-topic chatbot scope with soft redirect behavior."""

    def validate_topic(self, topic: Optional[str]) -> bool:
        return bool(topic) and topic in ALLOWED_TOPICS

    def infer_topic(self, message: str) -> Optional[str]:
        text = message.lower()
        scores = {topic: 0 for topic in ALLOWED_TOPICS}

        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[topic] += 1

        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0 else None

    def decide(self, message: str, requested_topic: Optional[str]) -> TopicDecision:
        cleaned = re.sub(r"\s+", " ", message.strip().lower())

        if self.validate_topic(requested_topic):
            # If topic is explicitly selected in UI/API, keep it unless message is empty.
            return TopicDecision(
                topic=requested_topic,  # type: ignore[arg-type]
                in_scope=bool(cleaned),
                redirected=False,
                reason="topic_selected",
            )

        inferred = self.infer_topic(cleaned)
        if inferred:
            return TopicDecision(
                topic=inferred,
                in_scope=True,
                redirected=False,
                reason="topic_inferred",
            )

        # Out-of-scope: soft redirect to a safe default.
        return TopicDecision(
            topic="brain_tumor",
            in_scope=False,
            redirected=True,
            reason="out_of_scope",
        )

    def build_redirect_reply(self, suggested_topic: str) -> str:
        topic_name = TOPIC_DISPLAY_NAMES.get(suggested_topic, "Brain Tumor")
        return (
            "I can help best with Brain Tumor, Pneumonia, or Fetal Ultrasound topics. "
            f"Let us continue with {topic_name}. Ask a question in that area, and I will help."
        )
