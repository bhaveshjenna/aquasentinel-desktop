"""Train the action-conditioned world model, federated across simulated ponds. Offline.

    python scripts/pretrain_world_model.py                  # federated + central + local
    python scripts/pretrain_world_model.py --rounds 4       # quick pass
    python scripts/pretrain_world_model.py --clients 6

Why the data is simulated, stated once here and repeated in the report: the real pond dataset
(4 catfish ponds, Nsukka) records water quality but no log of aerator, feeding or liming
commands - so it contains no ``(action -> response)`` pairs, and no amount of federation over
it can teach a model what an aerator does. The action-conditioned part has to be learned where actions exist, which is the
twin.

What is still real: the **encoder** is warm-started from ``forecaster_federated.pt``, the model
FedAvg produced across the real ponds, so the representation the planner's decision is
computed from is the federated one.

What "federated" means here: each client is a pond with its own perturbed physical parameters -
volume, depth, aeration efficiency, algal productivity, temperature regime - which is the
non-IID structure a real multi-farm deployment has. Clients exchange weights only, through the
same ``fedavg`` the forecaster uses.

Writes to models/: world_model_federated.pt, world_model_central.pt, world_model_local_<id>.pt
Writes to outputs/: world_model_clients.csv, world_model_rounds.csv, world_model_comparison.csv
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agent import actuators as ACT  # noqa: E402
from src.common.runtime import MODELS_DIR, OUTPUTS_DIR, load_config, set_seed  # noqa: E402
from src.common.schema import Action  # noqa: E402
from src.federated.simulate import fedavg  # noqa: E402
from src.perception.world_model import CHANNELS, ActionWorldModel, action_features  # noqa: E402
from src.twin.pond import PondTwin  # noqa: E402
from src.twin.scenarios import load_scenarios  # noqa: E402


# ======================================================================================
# Heterogeneous simulated ponds - the clients
# ======================================================================================
def pond_variants(n: int, seed: int = 7) -> List[Tuple[str, Dict[str, Any]]]:
    """`n` pond configs differing in the parameters that actually change the dynamics.

    These are the axes a real farm would differ on, not cosmetic jitter: a deeper pond
    stratifies and reaerates differently, a more productive pond swings harder overnight, a
    weaker aerator takes longer to recover. That heterogeneity is what makes averaging their
    weights a non-trivial thing to do.
    """
    rng = np.random.default_rng(seed)
    base = load_config("twin")
    out: List[Tuple[str, Dict[str, Any]]] = []
    for i in range(n):
        cfg = copy.deepcopy(base)
        cfg["pond"]["volume_m3"] = float(base["pond"]["volume_m3"] * rng.uniform(0.55, 1.8))
        cfg["pond"]["depth_m"] = float(base["pond"]["depth_m"] * rng.uniform(0.75, 1.45))
        cfg["oxygen"]["k_aerator_per_h"] = float(base["oxygen"]["k_aerator_per_h"] * rng.uniform(0.6, 1.4))
        cfg["oxygen"]["k_reaeration_per_h"] = float(base["oxygen"]["k_reaeration_per_h"] * rng.uniform(0.7, 1.5))
        cfg["oxygen"]["p_max_per_h"] = float(base["oxygen"]["p_max_per_h"] * rng.uniform(0.7, 1.45))
        cfg["temperature"]["mean_c"] = float(base["temperature"]["mean_c"] + rng.uniform(-3.0, 3.0))
        cfg["temperature"]["amplitude_c"] = float(base["temperature"]["amplitude_c"] * rng.uniform(0.7, 1.5))
        out.append((f"pond_{i:02d}", cfg))
    return out


# ======================================================================================
# Data generation: what does the pond do under a held action?
# ======================================================================================
def make_dataset(
    twin_cfg: Dict[str, Any],
    n_episodes: int,
    seed: int,
    window: int,
    horizon: int,
    scales: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the twin under randomised actions and cut (window, action, future) triples.

    The action is held for a random fraction of the horizon and then released, which is exactly
    what the planner does with `hold_action_steps` - a model trained only on "held for the whole
    horizon" would systematically over-predict what a candidate buys.
    """
    rng = np.random.default_rng(seed)
    scenarios = list(load_scenarios().values())
    candidates = ACT.candidates()
    Xs, As, Ys = [], [], []

    for ep in range(n_episodes):
        scn = scenarios[ep % len(scenarios)]
        twin = PondTwin(cfg=twin_cfg, seed=int(rng.integers(1, 10**6)),
                        start_hour_of_day=float(rng.uniform(0, 24)), scenario=scn)
        steps = window + horizon + 90
        hist: List[List[float]] = []

        # Warm the pond up under a random but coherent action stream, so the windows the model
        # sees at training time look like the windows it will see behind a working controller.
        act = Action.no_op()
        hold_left = 0
        for t in range(steps):
            if hold_left <= 0:
                act = candidates[int(rng.integers(0, len(candidates)))]
                hold_left = int(rng.integers(3, 30))
            hold_left -= 1
            st = twin.step(act)
            hist.append([getattr(st, c) for c in CHANNELS])

            # Once a full window exists, fork the pond and measure a candidate action properly.
            if t >= window - 1 and rng.random() < 0.5:
                w = np.asarray(hist[-window:], dtype=np.float32)
                cand = candidates[int(rng.integers(0, len(candidates)))]
                hold_steps = int(rng.integers(1, horizon + 1))
                sim = twin.copy()
                sim.scenario = None          # the planner cannot know the schedule; nor may this
                fut = []
                for k in range(horizon):
                    s2 = sim.step(cand if k < hold_steps else Action.no_op())
                    fut.append([getattr(s2, c) for c in CHANNELS])
                fut = np.asarray(fut, dtype=np.float32)

                last = w[-1]
                Xs.append(np.concatenate(
                    [(w - last) / scales, np.ones_like(w)], axis=1).astype(np.float32))
                As.append(action_features(cand, hold_steps / horizon))
                Ys.append(((fut - last[None, :]) / scales[None, :]).astype(np.float32))

        if verbose and (ep + 1) % 4 == 0:
            print(f"      episode {ep + 1}/{n_episodes}: {len(Xs)} samples")

    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(As, dtype=np.float32),
            np.asarray(Ys, dtype=np.float32))


# ======================================================================================
# Clients
# ======================================================================================
class PondWMClient:
    """One simulated pond. Owns its data and its model; only weights ever leave."""

    def __init__(self, client_id: str, twin_cfg, model_cfg, seed: int,
                 n_episodes: int, val_fraction: float = 0.15, verbose=False):
        self.client_id = client_id
        self.twin_cfg = twin_cfg
        self.model = ActionWorldModel(model_cfg, seed=seed)
        X, A, Y = make_dataset(twin_cfg, n_episodes, seed, self.model.window,
                               self.model.horizon, self.model.scales, verbose)
        n_val = int(len(X) * val_fraction)
        n_tr = len(X) - n_val
        self.X, self.A, self.Y = X[:n_tr], A[:n_tr], Y[:n_tr]
        self.Xv, self.Av, self.Yv = X[n_tr:], A[n_tr:], Y[n_tr:]

    @property
    def n_samples(self) -> int:
        return int(len(self.X))

    def local_train(self, epochs=None) -> Dict[str, float]:
        return self.model.fit(self.X, self.A, self.Y, epochs=epochs)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def evaluate_weights(self, w) -> Dict[str, float]:
        """RMSE on this client's own held-out tail, in scaled-delta units and in mg/L for DO."""
        if len(self.Xv) == 0:
            return {"rmse": float("nan"), "rmse_do_mgl": float("nan"), "n": 0}
        keep = self.model.get_weights()
        self.model.set_weights(w)
        import torch
        from src.common.runtime import device
        with torch.no_grad():
            pred = self.model.net(
                torch.from_numpy(self.Xv).to(device()),
                torch.from_numpy(self.Av).to(device()),
            ).cpu().numpy()
        self.model.set_weights(keep)
        err = pred - self.Yv
        do_i = CHANNELS.index("do")
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "rmse_do_mgl": float(np.sqrt(np.mean(err[:, :, do_i] ** 2)) * self.model.scales[do_i]),
            "n": int(len(self.Xv)),
        }


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8, help="twin episodes per client")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=2, help="ponds no regime trains on")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_config("models")
    wm_cfg = model_cfg.get("world_model", {})

    print("=" * 78)
    print("AquaSentinel - action-conditioned world model, federated across simulated ponds")
    print("=" * 78)
    print("  The real pond data logs no actuator commands, so action->response pairs come from the twin.")
    print("  The ENCODER is warm-started from the federated forecaster trained on the real ponds.\n")

    variants = pond_variants(args.clients + args.holdout, seed=args.seed)
    train_v, hold_v = variants[: args.clients], variants[args.clients :]

    print(f"Building {len(train_v)} training ponds + {len(hold_v)} held-out ponds ...")
    clients = []
    for i, (cid, cfg) in enumerate(train_v):
        c = PondWMClient(cid, cfg, model_cfg, seed=args.seed + i, n_episodes=args.episodes)
        clients.append(c)
        print(f"  {cid}: {c.n_samples} training samples, {len(c.Xv)} val")
    holdouts = []
    for i, (cid, cfg) in enumerate(hold_v):
        h = PondWMClient(cid, cfg, model_cfg, seed=args.seed + 500 + i, n_episodes=max(args.episodes // 2, 2))
        holdouts.append(h)
        print(f"  {cid}: HELD OUT, {len(h.Xv)} val")

    fed_path = MODELS_DIR / "forecaster_federated.pt"
    warm = bool(wm_cfg.get("warm_start_from_federated", True))

    def fresh() -> ActionWorldModel:
        m = ActionWorldModel(model_cfg, seed=args.seed)
        if warm:
            ok = m.warm_start_encoder(fed_path)
            if not ok:
                print("  ! encoder warm-start failed; falling back to random init")
        return m

    # ---------------------------------------------------------------- independent-local
    print("\n[independent-local] each pond trains alone, nothing shared")
    local_weights = {}
    for c in clients:
        m = fresh()
        c.set_weights(m.get_weights())
        info = c.local_train(epochs=args.rounds * args.local_epochs)
        local_weights[c.client_id] = c.get_weights()
        print(f"    {c.client_id:10s} n={c.n_samples:6d}  loss {info['loss']:.6f}")

    # ---------------------------------------------------------------- federated
    print(f"\n[federated] FedAvg, {args.rounds} rounds x {args.local_epochs} local epochs")
    global_w = fresh().get_weights()
    rounds_log = []
    for r in range(args.rounds):
        local, counts, losses = [], [], []
        for c in clients:
            c.set_weights(global_w)
            info = c.local_train(epochs=args.local_epochs)
            local.append(c.get_weights())
            counts.append(c.n_samples)
            losses.append(info["loss"])
        global_w = fedavg(local, counts)
        scored = [h.evaluate_weights(global_w)["rmse_do_mgl"] for h in holdouts]
        rounds_log.append({"round": r + 1, "mean_local_loss": float(np.mean(losses)),
                           "holdout_rmse_do_mgl": float(np.mean(scored))})
        print(f"    round {r + 1:2d}/{args.rounds}  loss {np.mean(losses):.6f}  "
              f"held-out DO RMSE {np.mean(scored):.4f} mg/L")

    # ---------------------------------------------------------------- central
    print("\n[central] pooled upper reference (violates the privacy premise)")
    cm = fresh()
    Xc = np.concatenate([c.X for c in clients]); Ac = np.concatenate([c.A for c in clients])
    Yc = np.concatenate([c.Y for c in clients])
    info = cm.fit(Xc, Ac, Yc, epochs=args.rounds * args.local_epochs)
    central_w = cm.get_weights()
    print(f"    pooled {len(Xc)} samples, loss {info['loss']:.6f}")

    # ---------------------------------------------------------------- score + save
    print("\nScoring on held-out ponds no regime has trained on ...")
    rows = []
    for h in holdouts:
        for name, w in (("federated", global_w), ("central", central_w)):
            m = h.evaluate_weights(w); m.update({"regime": name, "holdout_pond": h.client_id})
            rows.append(m)
        for cid, w in local_weights.items():
            m = h.evaluate_weights(w)
            m.update({"regime": "independent_local", "holdout_pond": h.client_id, "source_model": cid})
            rows.append(m)
    comp = pd.DataFrame(rows)
    summary = comp.groupby("regime")[["rmse", "rmse_do_mgl"]].mean().reset_index()
    print(summary.to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ActionWorldModel(model_cfg, seed=args.seed)
    out.set_weights(global_w); out.save(MODELS_DIR / "world_model_federated.pt")
    out.set_weights(central_w); out.save(MODELS_DIR / "world_model_central.pt")
    for cid, w in local_weights.items():
        out.set_weights(w); out.save(MODELS_DIR / f"world_model_local_{cid}.pt")

    comp.to_csv(OUTPUTS_DIR / "world_model_comparison.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "world_model_summary.csv", index=False)
    pd.DataFrame(rounds_log).to_csv(OUTPUTS_DIR / "world_model_rounds.csv", index=False)
    pd.DataFrame([{"client_id": c.client_id, "n_train": c.n_samples, "n_val": len(c.Xv),
                   "volume_m3": c.twin_cfg["pond"]["volume_m3"],
                   "depth_m": c.twin_cfg["pond"]["depth_m"],
                   "k_aerator_per_h": c.twin_cfg["oxygen"]["k_aerator_per_h"]}
                  for c in clients]).to_csv(OUTPUTS_DIR / "world_model_clients.csv", index=False)

    print(f"\nDone. models/world_model_federated.pt written; the planner will use it "
          f"(predictive.world_model: true).")


if __name__ == "__main__":
    main()
