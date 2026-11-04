"""Neural hybrids in PyTorch. All train on CPU in minutes; the data are a handful of packs, so every model is small.

  UDE          universal differential equation: the physics fade law times a learned correction exp(g(z)).
               Starts as pure physics (g = 0) and can never predict a capacity gain.
  GRU, Transformer  the same decrement form with a recurrent or attention memory of usage. `*_free` drops the physics scale.
  PINN         x(tau, Ah) network with the physics residual and monotonicity in the loss, by autodiff.
  CurveCNN     raw voltage-charge curve of one session -> capacity (session nowcast).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .physics import PhysicsODE, phi

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _standardise(train_mats):
    a = np.vstack(train_mats)
    m, s = np.nanmean(a, 0), np.nanstd(a, 0)
    return np.where(np.isfinite(m), m, 0), np.where(s > 1e-8, s, 1.0)


def _z(df, cols, m, s):
    return np.nan_to_num((df[cols].to_numpy(float) - m) / s, nan=0.0, posinf=0.0, neginf=0.0)


class MLP(nn.Module):
    def __init__(self, n, hidden, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class Causal(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.proj = nn.Linear(n, hidden)
        enc = nn.TransformerEncoderLayer(hidden, nhead=2, dim_feedforward=2 * hidden, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2, enable_nested_tensor=False)

    def forward(self, x):
        n = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=x.device)
        return self.enc(self.proj(x), mask=mask, is_causal=True)


class DecrementNet(nn.Module):
    """x_k = x_0 - cumulative sum of decrements. Decrements are positive by construction."""

    def __init__(self, kind, n, hidden, a, b, n_packs):
        super().__init__()
        self.kind = kind
        self.la, self.lb = nn.Parameter(torch.tensor(np.log(max(a, 1e-4)), dtype=torch.float32)), nn.Parameter(torch.tensor(np.log(max(b, 1e-4)), dtype=torch.float32))
        self.x0 = nn.Parameter(torch.ones(n_packs))
        if kind.startswith("gru"):
            self.rnn = nn.GRU(n, hidden, batch_first=True)
        elif kind.startswith("transformer"):
            self.rnn = Causal(n, hidden)
        else:
            self.rnn = None
        self.head = MLP(n if self.rnn is None else hidden, hidden)
        self.free = kind.endswith("free")

    def decrements(self, z, b1, b2):
        """z: (1, n, features); b1, b2: physics increments (1, n)."""
        if self.rnn is None:
            h = z
        elif self.kind.startswith("gru"):
            h, _ = self.rnn(z)
        else:
            h = self.rnn(z)
        g = self.head(h).squeeze(-1)
        phys = torch.exp(self.la) * b1 + torch.exp(self.lb) * b2
        if self.free:
            return nn.functional.softplus(g - 7.0) * (b1 + b2 > 0).float(), g
        return phys * torch.exp(g.clamp(-3, 3)), g


class DecrementForecaster:
    """Trains on the training packs' label sequences; predicts by rolling the decrements forward from the pack's causal state."""

    def __init__(self, kind, tau0, sessions, cols, cfg, seeds=None):
        self.kind, self.tau0, self.sessions, self.cols, self.c = kind, tau0, sessions, cols, cfg
        self.name = kind
        self.seeds = seeds or cfg["seeds"]

    def _tensors(self, sess):
        r, ah = phi(sess, self.tau0)
        b1, b2 = np.r_[0.0, np.diff(r)], np.r_[0.0, np.diff(ah)]
        z = _z(sess, self.cols, self.m, self.s)
        return (torch.tensor(z[None], dtype=torch.float32), torch.tensor(b1[None], dtype=torch.float32), torch.tensor(b2[None], dtype=torch.float32))

    def fit(self, train):
        c = self.c
        keys = list(train)
        seqs = []
        rows = []
        for k in keys:
            s = self.sessions[k]
            s = s[s.flight_id.isin(train[k].flight_id)]
            rows.append(s)
        self.m, self.s = _standardise([r[self.cols].to_numpy(float) for r in rows])
        for k, s in zip(keys, rows):
            h = train[k].set_index("flight_id").loc[s.flight_id]
            z, b1, b2 = self._tensors(s)
            y = torch.tensor(h.x_s.to_numpy(float), dtype=torch.float32)
            w = torch.tensor(1.0 / np.maximum(h.sd_s.to_numpy(float), 2e-3) ** 2, dtype=torch.float32)
            seqs.append((z, b1, b2, y, w / w.mean()))
        phys = PhysicsODE(self.tau0).fit(train)
        self.a0, self.b0 = phys.a, phys.b
        self.nets = []
        n_val = [max(5, int(0.15 * len(q[3]))) for q in seqs]
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = DecrementNet(self.kind, len(self.cols), c["hidden"], self.a0, self.b0, len(keys))
            opt = torch.optim.Adam(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
            best, best_state, wait = np.inf, None, 0
            for ep in range(c["epochs"]):
                opt.zero_grad()
                tr, va = 0.0, 0.0
                for i, (z, b1, b2, y, w) in enumerate(seqs):
                    dec, g = net.decrements(z, b1, b2)
                    pred = net.x0[i] - torch.cumsum(dec, 1).squeeze(0)
                    e = w * (pred - y) ** 2
                    tr = tr + e[:-n_val[i]].mean() + 1e-5 * (g ** 2).mean()
                    va = va + e[-n_val[i]:].mean().detach()
                tr.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if float(va) < best - 1e-9:
                    best, best_state, wait = float(va), {k_: v.clone() for k_, v in net.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait > c["patience"]:
                        break
            if best_state:
                net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, hist, fut):
        k = (hist.aircraft_label.iloc[0], int(hist.battery.iloc[0]))
        sess = self.sessions[k]
        t0 = hist.t_days.iloc[-1]
        z, b1, b2 = self._tensors(sess[sess.t_days <= fut.t_days.max()])
        tt = sess[sess.t_days <= fut.t_days.max()].t_days.to_numpy(float)
        preds = []
        for net in self.nets:
            dec, _ = net.decrements(z, b1, b2)
            cum = np.cumsum(dec.squeeze(0).numpy())
            at0 = np.interp(t0, tt, cum)
            preds.append(hist.x_f.iloc[-1] - (np.interp(fut.t_days.to_numpy(float), tt, cum) - at0))
        p = np.array(preds)
        return p.mean(0), np.hypot(p.std(0), hist.sd_f.iloc[-1])


class PINNNet(nn.Module):
    def __init__(self, n_extra, n_packs, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + n_extra, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.x0 = nn.Parameter(torch.ones(n_packs))
        self.la, self.lb = nn.Parameter(torch.tensor(-3.0)), nn.Parameter(torch.tensor(-4.0))

    def forward(self, r, ah, extra):
        return self.net(torch.cat([r, ah, extra], 1)).squeeze(1)


class PINN:
    """Data loss weighted by label uncertainty + physics residual by autodiff + monotonicity penalty."""
    name = "pinn"

    def __init__(self, tau0, sessions, cols, cfg, seeds=None):
        self.tau0, self.sessions, self.cols, self.c = tau0, sessions, cols, cfg
        self.seeds = seeds or cfg["seeds"]

    def fit(self, train):
        c = self.c
        keys = list(train)
        R, A, E, Y, W, P = [], [], [], [], [], []
        rows = []
        for i, k in enumerate(keys):
            s = self.sessions[k]
            s = s[s.flight_id.isin(train[k].flight_id)]
            rows.append(s)
        self.m, self.s = _standardise([r[self.cols].to_numpy(float) for r in rows])
        for i, (k, s) in enumerate(zip(keys, rows)):
            h = train[k].set_index("flight_id").loc[s.flight_id]
            r, ah = phi(s, self.tau0)
            R.append(r), A.append(ah), E.append(_z(s, self.cols, self.m, self.s)), Y.append(h.x_s.to_numpy(float))
            W.append(1.0 / np.maximum(h.sd_s.to_numpy(float), 2e-3) ** 2), P.append(np.full(len(s), i))
        to = lambda a: torch.tensor(np.concatenate(a) if isinstance(a, list) else a, dtype=torch.float32)
        r, a, e, y, w = to(R)[:, None], to(A)[:, None], torch.tensor(np.vstack(E), dtype=torch.float32), to(Y), to(W)
        pid = torch.tensor(np.concatenate(P), dtype=torch.long)
        w = w / w.mean()
        phys = PhysicsODE(self.tau0).fit(train)
        reach_r, reach_a = 0.6 * float(r.max() - r.min()), 0.6 * float(a.max() - a.min())
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = PINNNet(len(self.cols), len(keys), c["hidden"])
            with torch.no_grad():
                net.la.fill_(float(np.log(max(phys.a, 1e-4))))
                net.lb.fill_(float(np.log(max(phys.b, 1e-4))))
            opt = torch.optim.Adam(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
            for ep in range(c["epochs"]):
                opt.zero_grad()
                # the physics is enforced at collocation points that reach beyond the training range: that is what lets a PINN extrapolate
                i = torch.randint(0, len(r), (256,))
                rr = (r[i] + torch.rand(256, 1) * reach_r).requires_grad_(True)
                aa = (a[i] + torch.rand(256, 1) * reach_a).requires_grad_(True)
                fc = net(rr, aa, e[i])
                dr, da = torch.autograd.grad(fc.sum(), [rr, aa], create_graph=True)
                pred = net.x0[pid] + net(r, a, e)
                data = (w * (pred - y) ** 2).mean()
                res = ((dr + torch.exp(net.la)) ** 2).mean() + ((da + torch.exp(net.lb)) ** 2).mean()
                mono = (torch.relu(dr) ** 2).mean() + (torch.relu(da) ** 2).mean()
                loss = data + c["physics_weight"] * res + c["monotone_weight"] * mono
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, hist, fut):
        k = (hist.aircraft_label.iloc[0], int(hist.battery.iloc[0]))
        sess = self.sessions[k].set_index("flight_id")
        to = lambda a: torch.tensor(a, dtype=torch.float32)
        last = sess.loc[[hist.flight_id.iloc[-1]]].reset_index()
        fu = sess.loc[fut.flight_id].reset_index()
        preds = []
        with torch.no_grad():
            for net in self.nets:
                r0, a0 = phi(last, self.tau0)
                f0 = net(to(r0)[:, None], to(a0)[:, None], to(_z(last, self.cols, self.m, self.s)))
                r, a = phi(fu, self.tau0)
                f = net(to(r)[:, None], to(a)[:, None], to(_z(fu, self.cols, self.m, self.s)))
                preds.append(hist.x_f.iloc[-1] + (f - f0).numpy())
        p = np.array(preds)
        return p.mean(0), np.hypot(p.std(0), hist.sd_f.iloc[-1])


class CurveCNN(nn.Module):
    def __init__(self, ch, n_extra=0):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(ch, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.conv(x).squeeze(1)


class CurveNow:
    """Session curve -> capacity. X: (n, channels, bins); y: leave-one-out label, so the session's own reading is not in its target."""
    name = "curve_cnn"

    def __init__(self, cfg, seeds=None):
        self.c, self.seeds = cfg, seeds or cfg["seeds"]

    def fit(self, X, y, w):
        self.mean, self.std = float(np.mean(y)), float(np.std(y) + 1e-6)
        X = torch.tensor(X, dtype=torch.float32)
        yy = torch.tensor((y - self.mean) / self.std, dtype=torch.float32)
        w = torch.tensor(w / w.mean(), dtype=torch.float32)
        self.nets = []
        n = len(yy)
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = CurveCNN(X.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
            for ep in range(self.c["cnn_epochs"]):
                net.train()
                perm = torch.randperm(n)
                for i in range(0, n, 32):
                    b = perm[i:i + 32]
                    shift = int(torch.randint(-2, 3, (1,)))
                    opt.zero_grad()
                    loss = (w[b] * (net(torch.roll(X[b], shift, 2)) - yy[b]) ** 2).mean()
                    loss.backward()
                    opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.mean + self.std * np.mean([n(X).numpy() for n in self.nets], axis=0)
