"""JSONL-backed detection event storage."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class DetectionEventStore:
    """Append-only event store for recent detection records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "id": str(uuid4()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **event,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        lines = self.path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in reversed(lines[-max(limit, 1) :]):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def summary(self, limit: int = 500) -> dict[str, Any]:
        events = self.list_recent(limit=limit)
        prediction_counts = Counter(event.get("prediction", "Unknown") for event in events)
        risk_counts = Counter(event.get("risk_level", "Unknown") for event in events)
        attack_counts = Counter(event.get("attack_type", "unknown") for event in events)

        return {
            "total_events": len(events),
            "prediction_counts": dict(prediction_counts),
            "risk_counts": dict(risk_counts),
            "attack_type_counts": dict(attack_counts),
            "latest": events[0] if events else None,
        }

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")
