"""A ModelBundle = the production model for EVERY forecast variable.

Each variable can be served by a different family (e.g. LSTM for
temperature, Random Forest for AQI) - whichever validated best.
Saved as one folder in the Model Registry:
    bundle.json            metadata, metrics, reference stats for drift
    <var>.joblib           the fitted model object per variable
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.fuse_outputs import to_output_frame

OUTPUT_VARS_ORDER = ["aqi", "temp_c", "precip_mm", "rain_prob", "humidity_pct", "uvi", "wind_u", "wind_v"]


class ModelBundle:
    def __init__(self, meta: dict, models: dict):
        self.meta = meta
        self.models = models  # var -> ForecastModel

    @property
    def version(self):
        return self.meta.get("version")

    @classmethod
    def load(cls, folder: Path, version: int | None = None) -> "ModelBundle":
        folder = Path(folder)
        meta = json.loads((folder / "bundle.json").read_text())
        if version is not None:
            meta["version"] = version
        models = {var: joblib.load(folder / comp["file"]) for var, comp in meta["components"].items()}
        return cls(meta, models)

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for var, model in self.models.items():
            joblib.dump(model, folder / f"{var}.joblib", compress=3)
            self.meta["components"][var]["file"] = f"{var}.joblib"
        (folder / "bundle.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return folder

    def predict_raw(self, frame: pd.DataFrame, origin_pos: int) -> dict[str, np.ndarray]:
        idx = np.array([origin_pos])
        return {var: m.predict(frame, idx)[0] for var, m in self.models.items()}

    def forecast(self, frame: pd.DataFrame, origin_pos: int) -> pd.DataFrame:
        """24-row DataFrame with the six metric groups' output variables."""
        raw = self.predict_raw(frame, origin_pos)
        origin_ts = int(frame.loc[origin_pos, "ts"])
        return to_output_frame(raw, origin_ts)
