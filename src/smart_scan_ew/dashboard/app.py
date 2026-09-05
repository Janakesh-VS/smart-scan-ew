"""Phase 5 dashboard: Streamlit UI.

Contains NO simulation, scheduler, or evaluator logic of its own —
everything here calls `dashboard/controller.py`, which wraps the real
Phase 1-4 components and Phase 3 evaluator APIs unchanged. See
ARCHITECTURE.md's Phase 5 section for the full rationale and the
project owner's locked decisions.

Run with:
    streamlit run src/smart_scan_ew/dashboard/app.py

STREAMLIT SESSION-STATE NOTES (Part 13's explicit ask — read before
changing this file):
- The `DashboardController` instance lives in `st.session_state`, never
  a module-level global, so each browser session gets its own
  simulation and nothing leaks across users/tabs.
- The simulation is reset (a fresh environment/receiver/state/scheduler)
  ONLY when the resolved configuration tuple (scheduler, scenario,
  num_bands, seed, gamma, c) actually differs from
  `st.session_state.applied_config`, or when the operator presses
  "Reset" — never on an unrelated rerun. Clicking "Step Once", for
  example, does not touch any of those values, so no config-driven
  reset happens on that rerun.
- "Start"/"Pause" drive a bounded auto-advance loop (a fixed number of
  steps per rerun, a short sleep, then `st.rerun()`) — the standard,
  if imperfect, way to get a "live" feel under Streamlit's
  rerun-per-interaction model. This is best-effort, not a guaranteed
  smooth animation; "Step Once" is always available as the reliable,
  precise alternative for a live demo.
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from smart_scan_ew.dashboard.controller import (
    SCENARIO_PRESETS,
    SCHEDULER_DISPLAY_NAMES,
    SCHEDULER_NAMES,
    DashboardController,
    run_four_way_comparison,
)


def format_metric(value: float | None, digits: int = 3) -> str:
    """The ONLY formatting rule used anywhere in this file for a
    possibly-undefined evaluator metric: "N/A" for None, otherwise a
    fixed-precision number. NEVER substitutes 0 for None (Part 10's
    explicit requirement) — that would misrepresent "not enough data"
    as "measured zero"."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


st.set_page_config(page_title="Smart Scan EW Dashboard", layout="wide")

# ---- session state init (see module docstring) -------------------------
if "controller" not in st.session_state:
    st.session_state.controller = DashboardController()
if "running" not in st.session_state:
    st.session_state.running = False
if "applied_config" not in st.session_state:
    st.session_state.applied_config = None
if "ground_truth_history" not in st.session_state:
    st.session_state.ground_truth_history = []

controller: DashboardController = st.session_state.controller

st.title("Smart Scan EW — Live Dashboard")
st.caption("PS 26055 — ML-based Smart Scan Strategy for Electronic Warfare")

# ---- sidebar: configuration + controls ----------------------------------
with st.sidebar:
    st.header("Configuration")

    scheduler_display = st.selectbox(
        "Scheduler",
        options=[SCHEDULER_DISPLAY_NAMES[n] for n in SCHEDULER_NAMES],
    )
    scheduler_name = next(
        n for n in SCHEDULER_NAMES if SCHEDULER_DISPLAY_NAMES[n] == scheduler_display
    )

    scenario_display = st.selectbox("Scenario", options=list(SCENARIO_PRESETS.keys()))
    num_bands = st.slider("Number of bands", min_value=3, max_value=10, value=5)
    seed = st.number_input("Random / master seed", min_value=0, value=0, step=1)

    # Only exposed when relevant, per Part 9: "do not expose obscure
    # internal implementation variables."
    gamma, c = 0.95, 1.0
    if scheduler_name == "adaptive_ucb":
        st.subheader("Adaptive UCB parameters")
        gamma = st.slider("gamma (recency)", min_value=0.80, max_value=1.0, value=0.95, step=0.01)
        c = st.slider("c (exploration constant)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    speed = st.slider("Steps per tick (simulation speed)", min_value=1, max_value=20, value=1)

    desired_config = (scheduler_name, scenario_display, num_bands, int(seed), gamma, c)
    if st.session_state.applied_config != desired_config:
        controller.reset(
            scheduler_name=scheduler_name,
            num_bands=num_bands,
            seed=int(seed),
            emitter_specs=SCENARIO_PRESETS[scenario_display](num_bands),
            gamma=gamma,
            c=c,
        )
        st.session_state.applied_config = desired_config
        st.session_state.running = False
        st.session_state.ground_truth_history = []

    st.divider()
    col1, col2, col3 = st.columns(3)
    if col1.button("Start"):
        st.session_state.running = True
    if col2.button("Pause"):
        st.session_state.running = False
    if col3.button("Reset"):
        controller.reset(
            scheduler_name=scheduler_name,
            num_bands=num_bands,
            seed=int(seed),
            emitter_specs=SCENARIO_PRESETS[scenario_display](num_bands),
            gamma=gamma,
            c=c,
        )
        st.session_state.running = False
        st.session_state.ground_truth_history = []

    step_once = st.button("Step Once")


def _do_step() -> None:
    controller.step()
    # peek_ground_truth() is called HERE, in app.py, purely for the
    # debug visualization below — never inside controller.step() itself
    # and never fed back into it. See
    # tests/test_dashboard_ground_truth_isolation.py.
    st.session_state.ground_truth_history.append(controller.peek_ground_truth())


if step_once:
    _do_step()

if st.session_state.running:
    for _ in range(speed):
        _do_step()
    time.sleep(0.3)
    st.rerun()

# ---- tabs ---------------------------------------------------------------
tab_live, tab_perf, tab_compare, tab_demo = st.tabs(
    ["Live Simulation", "Performance", "Four-Way Comparison", "Demo Guide"]
)

with tab_live:
    st.subheader(
        f"Scheduler: {SCHEDULER_DISPLAY_NAMES[scheduler_name]}  |  t = {controller.current_time:g}"
    )

    history = controller.history
    if not history:
        st.info("Press **Step Once** or **Start** in the sidebar to begin the simulation.")
    else:
        latest = history[-1]

        # A. Time-frequency map. Ground truth is clearly labelled and
        # drawn from a history app.py collects itself via
        # peek_ground_truth() — never from the scheduler's decision path.
        st.markdown(
            "**GROUND TRUTH — EVALUATOR / DEBUG VIEW** "
            "_(never available to the scheduler)_"
        )
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for snapshot in st.session_state.ground_truth_history:
            for emitter in snapshot.emitters:
                if emitter.active:
                    ax.scatter(snapshot.time, emitter.band, color="lightcoral", s=15, zorder=1)
        times = [r.time for r in history]
        bands = [r.band for r in history]
        colors = ["seagreen" if r.observation.detected else "gray" for r in history]
        ax.scatter(
            times, bands, color=colors, s=40, marker="x", zorder=2,
            label="Receiver scan (green=HIT, gray=MISS)",
        )
        ax.set_xlabel("Simulation time")
        ax.set_ylabel("Band")
        ax.set_yticks(range(controller.num_bands))
        ax.legend(loc="upper right", fontsize="x-small")
        st.pyplot(fig)
        plt.close(fig)

        # B/C. Receiver position + latest observation
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Selected band", latest.band)
        col_b.metric("Latest observation", "HIT" if latest.observation.detected else "MISS")
        measured_power = latest.observation.info.get("measured_power")
        col_c.metric(
            "Measured power",
            f"{measured_power:.3f}" if measured_power is not None else "N/A",
        )

        # D. Adaptive UCB info + "why this band" explanation — entirely
        # from AdaptiveUcbScheduler.get_diagnostics(), no recomputation.
        if latest.ucb_diagnostics is not None:
            st.markdown("**Adaptive UCB state** (from `AdaptiveUcbScheduler.get_diagnostics()`)")
            ucb_rows = [
                {
                    "band_id": d.band_id,
                    "S_b": round(d.discounted_successes, 3),
                    "N_b": round(d.discounted_observations, 3),
                    "observed": d.observed,
                    "estimated_hit_rate": format_metric(d.estimated_hit_rate),
                    "exploration_bonus": format_metric(d.exploration_bonus),
                    "ucb_score": format_metric(d.ucb_score),
                    "selected_this_step": d.band_id == latest.band,
                }
                for d in latest.ucb_diagnostics
            ]
            st.dataframe(pd.DataFrame(ucb_rows), hide_index=True, width="stretch")

            reason = latest.decision_reason
            st.markdown("**Why this band was selected**")
            if reason.was_unobserved_band:
                st.code(f"Selected Band: B{reason.band}\nReason: previously unobserved band")
            else:
                st.code(
                    f"Selected Band: B{reason.band}\n"
                    f"Reason:\n"
                    f"    UCB Score = {format_metric(reason.ucb_score)}\n"
                    f"    Estimated Hit Rate = {format_metric(reason.estimated_hit_rate)}\n"
                    f"    Exploration Bonus = {format_metric(reason.exploration_bonus)}"
                )

        # E. Phase 2 belief state — from the real SimpleBeliefState.
        st.markdown("**Phase 2 belief state** (from `SimpleBeliefState.get_features()`)")

        def _last_detected_label(value: bool | None) -> str:
            # Kept as a single string type across all rows -- mixing
            # bool and the string "N/A" in one pandas column breaks
            # Arrow serialization in st.dataframe (caught by
            # tests/test_dashboard_app.py's headless smoke test).
            if value is None:
                return "N/A"
            return "HIT" if value else "MISS"

        belief_rows = [
            {
                "band_id": b.band_id,
                "observation_count": b.observation_count,
                "hit_count": b.hit_count,
                "estimated_probability": format_metric(b.estimated_probability),
                "last_detected": _last_detected_label(b.last_detected),
                "time_since_last_observed": format_metric(b.time_since_last_observed, digits=1),
            }
            for b in latest.belief.bands
        ]
        st.dataframe(pd.DataFrame(belief_rows), hide_index=True, width="stretch")

with tab_perf:
    st.subheader("Performance (real Phase 3 evaluator)")
    st.caption(
        "Runs a fresh experiment via `run_experiment_for_scheduler()`, using this "
        "session's exact configuration (same seed, scenario, scheduler, and "
        "hyperparameters) — not a re-display of the live steps above."
    )
    default_steps = len(controller.history) if controller.history else 50
    eval_steps = st.number_input("Steps to evaluate", min_value=1, value=default_steps)

    if st.button("Compute Performance Metrics"):
        result = controller.replay_metrics(num_steps=int(eval_steps))
        metric_rows = [
            ("Probability of Detection (Pd)", format_metric(result.probability_of_detection)),
            ("Probability of False Alarm (Pfa)", format_metric(result.probability_of_false_alarm)),
            ("Active-emitter interception rate", format_metric(result.interception_rate_active_emitters)),
            ("All-emitter interception rate", format_metric(result.interception_rate_all_emitters)),
            ("Average intercept time", format_metric(result.average_intercept_time)),
            ("Intercept time error", format_metric(result.average_intercept_time_error)),
            ("Average reward", format_metric(result.average_reward)),
            ("Average cost", format_metric(result.average_cost)),
        ]
        st.table(pd.DataFrame(metric_rows, columns=["Metric", "Value"]))

with tab_compare:
    st.subheader("Four-way comparison (real evaluator — no precomputed results)")
    n_seeds = st.slider("Number of trial seeds", min_value=2, max_value=20, value=5)
    base_seed = st.number_input("Base seed for trials", min_value=0, value=100, step=1)
    compare_steps = st.number_input("Steps per trial", min_value=10, value=100, step=10)

    if st.button("Run Four-Way Comparison"):
        seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
        with st.spinner("Running real experiments via run_repeated_trials()..."):
            results = run_four_way_comparison(
                num_bands=num_bands,
                num_steps=int(compare_steps),
                master_seeds=seeds,
                gamma=gamma,
                c=c,
                emitter_specs=SCENARIO_PRESETS[scenario_display](num_bands),
            )

        rows = [
            {
                "Scheduler": SCHEDULER_DISPLAY_NAMES[name],
                "Active IR": format_metric(results[name].interception_rate_active_emitters.mean),
                "Avg Intercept Time": format_metric(results[name].average_intercept_time.mean),
                "Pd": format_metric(results[name].probability_of_detection.mean),
                "Pfa": format_metric(results[name].probability_of_false_alarm.mean),
            }
            for name in SCHEDULER_NAMES  # fixed order — never sorted by performance
        ]
        st.table(pd.DataFrame(rows))

        fig, ax = plt.subplots(figsize=(6, 3))
        names = [SCHEDULER_DISPLAY_NAMES[n] for n in SCHEDULER_NAMES]
        values = [results[n].interception_rate_active_emitters.mean or 0.0 for n in SCHEDULER_NAMES]
        ax.bar(names, values, color="steelblue")  # identical color/style for every bar — no bias
        ax.set_ylabel("Active-emitter interception rate (mean)")
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        plt.close(fig)

with tab_demo:
    st.subheader("2-3 minute demo workflow")
    demo_steps = [
        "Select a scenario in the sidebar.",
        "Select **Round Robin** and press **Start** — watch the receiver scan evenly.",
        "Point out HIT/MISS in the Latest Observation panel.",
        "Open **Performance** and click **Compute Performance Metrics** for Round Robin.",
        "Switch the sidebar scheduler to **Adaptive UCB** (this resets the session).",
        "Press **Step Once** a few times — watch the per-band S_b/N_b/UCB table change.",
        "Point out the **Why this band was selected** explanation.",
        "Open **Four-Way Comparison** and click **Run Four-Way Comparison**.",
    ]
    st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(demo_steps, start=1)))
