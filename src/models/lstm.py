"""Deep learning model: LSTM (PyTorch).

Input:  the last `window_hours` (48) hours of a compact feature set
        (all target variables + calendar/solar features), standardised.
Output: the next 24 hourly values of ONE variable (direct multi-horizon).
Early stopping uses the last 10% of the TRAINING period (never the
validation set used for model comparison).
"""
from __future__ import annotations

import numpy as np

from src.models.base import ForecastModel
from src.utils.config import horizon, settings, target_vars

SEQ_EXTRA = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_elev_deg"]


def _torch():
    import torch
    from torch import nn

    return torch, nn


def _build_net(n_in: int, hidden: int, n_out: int):
    torch, nn = _torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return Net()


class LSTMModel(ForecastModel):
    family = "deep_learning"
    name = "lstm"

    def __init__(self, var: str):
        super().__init__(var)
        self.cfg = dict(settings()["training"]["lstm"])
        self.cols = list(dict.fromkeys(target_vars() + SEQ_EXTRA))
        self.state = None
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = 0.0

    def _windows(self, frame, idx):
        W = self.cfg["window_hours"]
        X = frame[self.cols].to_numpy(dtype=float)
        X = (X - self.x_mean) / self.x_std
        X = np.nan_to_num(X, nan=0.0)  # missing -> training mean
        out = np.zeros((len(idx), W, X.shape[1]), dtype=np.float32)
        for r, i in enumerate(idx):
            lo = i - W + 1
            seg = X[max(lo, 0):i + 1]
            out[r, W - len(seg):] = seg
        return out

    def fit(self, data, train_idx):
        torch, nn = _torch()
        seed = settings()["training"]["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_rows = data.frame.loc[: int(train_idx.max()), self.cols].to_numpy(dtype=float)
        self.x_mean = np.nanmean(train_rows, axis=0)
        self.x_std = np.nanstd(train_rows, axis=0) + 1e-6
        y = data.y(self.var, train_idx)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-6)

        n_es = max(int(len(train_idx) * 0.1), 1)
        fit_idx, es_idx = train_idx[:-n_es], train_idx[-n_es:]
        Xf = torch.tensor(self._windows(data.frame, fit_idx))
        yf = torch.tensor(((data.y(self.var, fit_idx) - self.y_mean) / self.y_std).astype(np.float32))
        Xe = torch.tensor(self._windows(data.frame, es_idx))
        ye = torch.tensor(((data.y(self.var, es_idx) - self.y_mean) / self.y_std).astype(np.float32))

        net = _build_net(len(self.cols), self.cfg["hidden_size"], horizon())
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg["learning_rate"])
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        bs = self.cfg["batch_size"]
        for _epoch in range(self.cfg["epochs"]):
            net.train()
            perm = torch.randperm(len(Xf))
            for b in range(0, len(Xf), bs):
                sel = perm[b:b + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xf[sel]), yf[sel])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(Xe), ye))
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.state = best_state or net.state_dict()
        return self

    def predict(self, frame, idx):
        torch, _ = _torch()
        net = _build_net(len(self.cols), self.cfg["hidden_size"], horizon())
        net.load_state_dict(self.state)
        net.eval()
        with torch.no_grad():
            out = net(torch.tensor(self._windows(frame, idx))).numpy()
        return self.postprocess(out * self.y_std + self.y_mean, frame, idx)
