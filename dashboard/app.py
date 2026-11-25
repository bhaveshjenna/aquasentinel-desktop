"""AquaSentinel dashboard - the live deliverable.

    streamlit run dashboard/app.py

Fully offline. Nothing here touches the network: the twin, the pretrained weights and the
cached real-data manifest all live on disk. If ``models/`` is empty the app still runs, with
the perception layer degraded to the rule-based risk function and no forecast overlay - the
Phase 2/3 experience - and says so in the sidebar rather than failing.
"""

from __future__ import annotations

import os
import sys
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from src.common.runtime import MODELS_DIR, load_config, set_seed  # noqa: E402
from src.agent.loop import AgentRun, Perception  # noqa: E402
from src.twin.scenarios import load_scenarios  # noqa: E402

from panels import agent as agent_panel  # noqa: E402
from panels import counterfactual as cf_panel  # noqa: E402
from panels import regime as regime_panel  # noqa: E402
from panels import risk_shap as risk_panel  # noqa: E402
from panels import scorecard as scorecard_panel  # noqa: E402
from panels import timeseries as ts_panel  # noqa: E402

st.set_page_config(page_title="Safe Predictive Control and Federated Forecasting for Aquaculture Water Quality Simulation", page_icon="🐟", layout="wide")

#: Ceiling on playback frames per second. Each frame is a full Streamlit rerun, and the
#: browser has to redraw nine plots inside it; pushing frames faster than it can draw them
#: only backs the websocket up, which is what makes playback appear to stall and then
#: resume hours of simulated time later.
MAX_FPS = 4.0

VIEWS = ["Live", "Counterfactual", "Agent", "Scorecard"]

CONTROLLERS = {
    "predictive": "Agent — predictive (learned MPC)",
    "reactive": "Agent — reactive rules",
    "static_threshold": "Baseline — static DO threshold",
    "do_nothing": "Baseline — do nothing",
    "oracle": "Reference — oracle (reads latent state)",
}


# ======================================================================================
# Model loading
# ======================================================================================
@st.cache_resource(show_spinner=False)
def load_models(model_source: str) -> Dict[str, Any]:
    """Load the pretrained stack for one weight source. Missing pieces degrade, never crash."""
    from src.perception.explain import RiskExplainer
    from src.perception.forecaster import LSTMForecaster
    from src.perception.regimes import RegimeModel
    from src.perception.risk import RiskModel

    out: Dict[str, Any] = {"forecaster": None, "regimes": None, "risk": None,
                           "explainer": None, "notes": []}

    path = {"federated": MODELS_DIR / "forecaster_federated.pt",
            "central": MODELS_DIR / "forecaster_central.pt"}.get(model_source)
    if path is None:  # independent-local: any one client's solo model
        locals_ = sorted(MODELS_DIR.glob("forecaster_local_*.pt"))
        path = locals_[0] if locals_ else None
        if path is not None:
            out["notes"].append(f"independent-local model: {path.stem[len('forecaster_local_'):]}")

    if path is not None and path.exists():
        try:
            out["forecaster"] = LSTMForecaster.load(path)
        except Exception as exc:
            out["notes"].append(f"forecaster failed to load: {exc}")
    else:
        out["notes"].append("no forecaster weights — run scripts/pretrain_federated.py")

    rp = MODELS_DIR / "regimes.pkl"
    if rp.exists():
        try:
            out["regimes"] = RegimeModel.load(rp)
        except Exception as exc:
            out["notes"].append(f"regime model failed to load: {exc}")

    kp = MODELS_DIR / "risk_model.pkl"
    if kp.exists():
        try:
            out["risk"] = RiskModel.load(kp)
            out["explainer"] = RiskExplainer(out["risk"])
        except Exception as exc:
            out["notes"].append(f"risk model failed to load: {exc}")
    else:
        out["notes"].append("no risk model — falling back to the rule-based risk bands")
    return out


@st.cache_data(show_spinner=False)
def data_summary() -> Optional[Dict[str, Any]]:
    try:
        from src.data import registry as R

        return R.summary()
    except Exception:
        return None


def available_sources() -> List[str]:
    out = []
    if (MODELS_DIR / "forecaster_federated.pt").exists():
        out.append("federated")
    if list(MODELS_DIR.glob("forecaster_local_*.pt")):
        out.append("independent-local")
    if (MODELS_DIR / "forecaster_central.pt").exists():
        out.append("central")
    return out or ["federated"]


# ======================================================================================
# Run lifecycle
# ======================================================================================
def new_run(scenario_id: str, controller: str, seed: int, model_source: str, guardrails: bool) -> AgentRun:
    set_seed(seed)
    models = load_models(model_source)
    perception = Perception(
        forecaster=models["forecaster"],
        regimes=models["regimes"],
        risk=models["risk"],
        explainer=models["explainer"],
    )
    return AgentRun(
        scenario_id,
        controller=controller,
        seed=seed,
        perception=perception,
        guardrails_enabled=guardrails,
    )


def ensure_run() -> AgentRun:
    s = st.session_state
    key = (s.scenario_id, s.controller, s.seed, s.model_source, s.guardrails)
    if s.get("_run_key") != key or "run" not in s:
        s.run = new_run(*key)
        s._run_key = key
        s.playing = False
        s.regime_history = []
    return s.run


def advance(run: AgentRun, n: int) -> None:
    for _ in range(n):
        if run.done():
            st.session_state.playing = False
            break
        d = run.tick()
        st.session_state.regime_history.append(
            {"t_hours": d.t_hours, "regime": d.regime, "anomaly": d.anomaly}
        )


# ======================================================================================
# Sidebar
# ======================================================================================
def sidebar() -> None:
    s = st.session_state
    scenarios = load_scenarios()

    st.sidebar.title("🐟 Safe Predictive Control and Federated Forecasting for Aquaculture Water Quality Simulation")
    st.sidebar.caption("Aquaculture water-quality monitoring and action suggestion")

    sid = st.sidebar.selectbox(
        "Scenario",
        list(scenarios),
        format_func=lambda k: scenarios[k].label,
        index=list(scenarios).index(s.scenario_id) if s.get("scenario_id") in scenarios else 1,
    )
    s.scenario_id = sid
    st.sidebar.caption(scenarios[sid].description)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Controller")
    s.agent_on = st.sidebar.toggle("Agent enabled", value=s.get("agent_on", True))
    if s.agent_on:
        options = ["predictive", "reactive"]
        s.controller = st.sidebar.radio(
            "Planner", options,
            index=options.index(s.controller) if s.get("controller") in options else 0,
            format_func=lambda k: CONTROLLERS[k],
        )
    else:
        options = ["do_nothing", "static_threshold", "oracle"]
        s.controller = st.sidebar.radio(
            "Baseline", options,
            index=options.index(s.controller) if s.get("controller") in options else 0,
            format_func=lambda k: CONTROLLERS[k],
        )
    s.guardrails = st.sidebar.toggle(
        "Guardrails active", value=s.get("guardrails", True),
        help="Hard safety rules. Turning them off is an ablation - the unsafe-applied counter "
             "is expected to leave zero.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Perception weights")
    sources = available_sources()
    s.model_source = st.sidebar.radio(
        "Model source", sources,
        index=sources.index(s.model_source) if s.get("model_source") in sources else 0,
        help="Which forecaster drives the agent: the federated global model, a single site's "
             "solo model, or the centrally-pooled reference.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Playback")
    # A rate, not a batch size. "Ticks per refresh" was only a real speed if every refresh
    # took the same time, and they do not - so the run drifted ahead of the displayed clock.
    s.speed = st.sidebar.select_slider(
        "Ticks per second", options=[1, 2, 4, 8, 16, 32], value=s.get("speed", 8)
    )
    sample_min = float(load_config("sensors")["sample_every_minutes"])
    st.sidebar.caption(f"≈ {s.speed * sample_min / 60:.1f} simulated hours per real second.")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("▶ Play" if not s.get("playing") else "⏸ Pause", width="stretch"):
        s.playing = not s.get("playing", False)
        s._last_frame = None  # start pacing from now, not from whenever playback last ran
    if c2.button("Step", width="stretch"):
        s.playing = False
        stepped = ensure_run()
        advance(stepped, stepped.ticks_per_decision)
    if c3.button("Reset", width="stretch"):
        s.playing = False
        s.pop("_run_key", None)
        s.regime_history = []

    s.seed = st.sidebar.number_input("Seed", value=int(s.get("seed", scenarios[sid].seed)), step=1)
    s.show_truth = st.sidebar.toggle(
        "Show twin ground truth", value=s.get("show_truth", False),
        help="Simulation only. A real farm sees the probe trace, never this.",
    )

    st.sidebar.markdown("---")
    models = load_models(s.model_source)
    for note in models["notes"]:
        st.sidebar.caption(f"⚠ {note}")
    info = data_summary()
    if info:
        st.sidebar.caption(
            f"Training data: {info['n_sites']} real fish ponds, {info['n_rows']:,} rows "
            f"({', '.join(f'{k}:{v}' for k, v in info['sources'].items())})."
        )
    st.sidebar.caption(
        "**The pond being controlled is a simulation.** The perception models are trained on "
        "real catfish-pond sensor data from Nsukka, Nigeria; every number here is relative, "
        "not a real-world guarantee."
    )


# ======================================================================================
# Main
# ======================================================================================
def main() -> None:
    s = st.session_state
    s.setdefault("scenario_id", "night_do_crash")
    s.setdefault("controller", "predictive")
    s.setdefault("model_source", available_sources()[0])
    s.setdefault("guardrails", True)
    s.setdefault("seed", 1002)
    s.setdefault("regime_history", [])
    s.setdefault("playing", False)

    sidebar()
    run = ensure_run()
    scenario = run.scenario
    bands = load_config("twin")["risk_bands"]

    from panels.common import clock

    # Advance *before* rendering, by however many ticks real elapsed time is worth. The old
    # loop stepped a fixed batch after drawing, so the panels always showed the previous
    # frame and the run went as fast as the server could manage rather than as fast as asked.
    frame_start = time.perf_counter()
    if s.playing and not run.done():
        last = s.get("_last_frame")
        elapsed = (frame_start - last) if last else 1.0 / MAX_FPS
        s._last_frame = frame_start
        # Never leap more than a second of simulated time in one frame: a slow frame must not
        # be repaid by skipping the pond forward past the excursion you were watching for.
        advance(run, max(1, min(int(round(elapsed * s.speed)), int(s.speed))))

    t_now = run.result.states[-1].t_hours if run.result.states else 0.0
    head = st.columns([3, 1, 1, 1])
    head[0].markdown(f"### {scenario.label}")
    head[1].metric("Simulated clock", clock(t_now, scenario.start_hour_of_day))
    head[2].metric("Progress", f"{run.tick_index}/{run.n_ticks}")
    head[3].metric("Controller", CONTROLLERS[run.controller].split("—")[-1].strip())

    if run.done():
        st.success("Scenario complete. Change the controller or the model source and reset to compare.")

    decision = run.result.decisions[-1] if run.result.decisions else None
    truth = run.result.frame()
    observed = run.stream.frame()

    # Not st.tabs: every tab body of an st.tabs runs on every rerun, so during playback the
    # counterfactual charts, the intervention log and the metrics table were all being rebuilt
    # and shipped four times a second while nobody was looking at them. This renders one.
    view = st.segmented_control(
        "View", VIEWS, default="Live", key="view", label_visibility="collapsed"
    ) or "Live"

    if view == "Live":
        scorecard_panel.render(run.result.metrics() if run.result.states else {})
        st.markdown("#### Live channels, forecast overlay and risk bands")
        ts_panel.render(
            truth, observed,
            forecast=run.last_perception.get("forecast"),
            bands=bands,
            show_truth=s.show_truth,
        )
        st.markdown("#### Regime and anomaly")
        regime_panel.render(
            run.last_perception.get("regime_name", "warming up"),
            run.last_perception.get("regime", -1),
            bool(run.last_perception.get("anomaly", False)),
            s.regime_history,
        )
        st.markdown("#### Risk and its drivers")
        risk_panel.render(
            run.last_perception.get("risk_class", "safe"),
            run.last_perception.get("risk_score", 0.0),
            decision.shap_top if decision else None,
        )

    elif view == "Counterfactual":
        st.markdown("#### Agent vs. an identical pond with the agent switched off")
        cf_panel.render(truth, run.result.shadow_frame(), bands=bands, channels=["do", "nh3"])
        st.caption(
            "Both ponds were forked at t=0 from the same state with the same seed, and receive "
            "the same perturbation schedule and the same sensor-noise realisation. The only "
            "difference between them is the agent."
        )

    elif view == "Agent":
        agent_panel.render(decision, run.result.decisions)

    elif view == "Scorecard":
        if run.result.states:
            m = run.result.metrics()
            scorecard_panel.render(m)
            st.markdown("#### Full metrics")
            st.dataframe(
                {"metric": list(m.keys()), "value": [str(v) for v in m.values()]},
                width="stretch", hide_index=True, height=420,
            )
        else:
            st.info("Run the scenario to populate the scorecard.")

    if s.playing and not run.done():
        # Hold the frame rate down to something the browser can actually paint. Without this
        # the server reruns flat out, the websocket backs up, and the first frame you see is
        # already many simulated hours past where you pressed play.
        time.sleep(max(0.0, 1.0 / MAX_FPS - (time.perf_counter() - frame_start)))
        st.rerun()


if __name__ == "__main__":
    main()
