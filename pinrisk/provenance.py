"""Provenance registry — every data layer declares where it came from.

Why this exists (and is non-negotiable): an insurer evaluating PinRisk will
ask, for every number, "based on what?". The registry is a small JSON file,
written when data is acquired and carried through to the final per-pincode
outputs, that answers exactly that — including the honest flag
`is_synthetic: true` while a layer is still sample data.

The dashboard uses it to (a) show sources per pincode and (b) display a large
warning banner whenever any core layer is synthetic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LayerProvenance:
    layer: str                 # e.g. "dem"
    source: str                # e.g. "FABDEM v1.2" or "SYNTHETIC sample"
    native_resolution: str     # e.g. "~30 m"
    is_synthetic: bool
    url: str = ""
    notes: str = ""            # caveats, processing applied, etc.
    swap_instructions: str = ""  # exactly how to replace with the real source


@dataclass
class ProvenanceRegistry:
    layers: dict = field(default_factory=dict)

    def add(self, p: LayerProvenance) -> None:
        self.layers[p.layer] = asdict(p)

    def any_synthetic(self) -> bool:
        return any(v["is_synthetic"] for v in self.layers.values())

    def synthetic_layers(self) -> list[str]:
        return [k for k, v in self.layers.items() if v["is_synthetic"]]

    def sources_summary(self) -> str:
        """One-line, human-readable source list (attached to each pincode)."""
        parts = []
        for name, v in self.layers.items():
            tag = " [SYNTHETIC]" if v["is_synthetic"] else ""
            parts.append(f"{name}: {v['source']}{tag}")
        return "; ".join(parts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.layers, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ProvenanceRegistry":
        reg = cls()
        reg.layers = json.loads(Path(path).read_text())
        return reg
