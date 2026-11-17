# Physics SOH label and hybrid models

`eplanes/soh/` builds a state-of-health (SOH) label from voltage and current alone, tests it, and then trains models to predict it.
No health number in this pipeline is calculated from the BMS. Reported SOH appears only as something to compare against, and reported
SOC only as a flag that a session was not logged between two others.

## Why not use the reported SOH, or capacity over an SOC window

Reported SOH jumps, so a model trained on it learns the estimator's revisions as well as the battery's ageing.
Capacity over an SOC window looks independent but is not: `Ah / (change in SOC)` uses the BMS's SOC, and the BMS scales SOC by its own
capacity estimate. When the BMS revises its capacity, this "measurement" moves with it. Any independent measure must be anchored on
something the BMS does not control. **Voltage is that anchor.**

## The physics

A cell's rest voltage is a fixed function of how full it is. How full it is depends on the charge moved and the capacity:

```
v = U(theta)        theta = theta_0 + Q / C
```

`U` is a property of the chemistry and is shared by the fleet. `Q` is integrated current. Turn the equation around and every session's
charge-versus-voltage curve is a scaled copy of one reference curve:

```
Q_k(v) = c_k + s_k * q_ref(v)        (1)

SOH_k = 100 * s_k / s_baseline
```

`c_k` only says how full the pack was at the start. `s_k` is the capacity. In plain words: **a healthy battery needs more amp-hours to
climb the same voltage range.** That is the whole idea, and no SOC is involved. It also works for partial charges, because equation (1)
needs only a part of the curve.

The voltage has to be the rest voltage, so the resistive part is removed first:

```
v_rest = v_terminal + I * r(T)        r(T) = r25 * exp(E * (1/T - 1/298.15))
```

`r25` comes from the charger's step-down, where the state of charge is the same on both sides of the step, so
`r = dV / dI` is purely resistive. `E` is fitted within pack and quarter, because season and age are otherwise confounded.

### Three independent capacity readings

| Name | Data | Equation |
|---|---|---|
| `chg` | Charge curve before the charger steps down | (1) |
| `dis` | Steady in-flight load | (1). Diagnostic only: on real data it rises over time, so it is contaminated by resistance growth |
| `r2r` | Rest voltage before and after a session, with the Ah moved between them | `s = Ah / [f(v_after) - f(v_before)]` |

The label combines `chg` and `r2r` (config `label.modalities`). Readings are combined as measured, with no rescaling between types;
their disagreement goes into the uncertainty. `r2r` pairs after a charge need at least 4 hours of rest, because the voltage is still relaxing.

`f` in `r2r` is not assumed. It is learned from the data by alternating between `f` and `s` (non-negative least squares).

Two more readings use the highest and lowest cell channels instead of the pack mean. They give a weakest-cell view that can be compared
with the BMS's weakest-cell SOH.

### The label: a state-space smoother

Single readings are noisy and sessions are uneven in time, so the label is the state of a small degradation model:

```
x[k+1] = x[k] - a * [sqrt(tau[k+1]) - sqrt(tau[k])] - b * dAh[k] + w[k]        w ~ N(0, q * dt)
y[m,k] = x[k] + beta' z[m,k] + e[m,k]
```

The first term is calendar fade, the second cycling fade. `z` holds temperature and the voltage window of the reading.
A Kalman filter and RTS smoother solve it, with Student-t weights so one bad session cannot pull the label. `a`, `b` and `q` are
fitted by maximum likelihood on the weighted readings, and `q` is capped (`label.q_max_per_day`), because outliers otherwise drive it to
its limit and the label follows noise.

* **Gap-aware.** Process noise grows with time, so the state may move further across a long idle period.
* **Fade is a prior, not a constraint.** If readings show a sustained rise the label follows and the step is listed in
  `L2_detected_steps.csv`. A pack's usable capacity can truly rise (balancing, a replaced module), so whether a rise is real is a question for
  the maintenance log. `soh_iso` gives the monotone version as a sensitivity check.
* **Three versions per session.** Smoothed (`soh_phys`), causal (`soh_filt`, uses only the past), and leave-one-out (`soh_loo`, excludes that
  session's own readings, closed form).
* **Honest uncertainty.** `soh_sd` adds the disagreement between reading types to the Kalman variance, because Kalman variance knows only about
  random noise.

Hyperparameters and reference curves come from the fit fleet only (config: `label.fit_fleet`); other aircraft are passed through the fitted model.

## How do we know the label is right?

There is no reference test in the data, so `validate` runs tests that need no ground truth:

| Test | Question | A good label shows |
|---|---|---|
| V1 BMS step | At a large BMS jump, did capacity move too? | slope near 0 (SOC-window near 1) |
| V2 Twin packs | Do two packs on one aircraft track each other? | smaller gap than the BMS |
| V3 Modalities | Do `chg`, `dis`, `r2r` give the same fade? | similar rates |
| V4 Voltage prediction | Using capacity known before a session, predict the rest voltage after it | lower error than reported SOH |
| V5 Coulomb closure | Charge in over charge out, per cycle | near 1 and no drift (a drifting sensor would look like fade) |
| V6 Curve shape | Do the lower and upper half of the voltage window fade alike? | small gap |

**V4 is the strongest test.** It asks the capacity estimate to predict a real, measured voltage it has not seen.

## Prediction

Two tasks:

* **Forecast:** at an origin, given the causal state and history, predict SOH 30, 90 and 180 days ahead, for the usage that actually followed.
* **Nowcast:** from one session's features, estimate SOH now.

| Family | Models |
|---|---|
| Baselines | `persistence`, `fleet_slope`, `linear_pack` |
| Physics | `physics_ode` (calendar + cycling law), `physics_bayes` (pack-specific rate that starts at the fleet law) |
| Hybrid tabular | `residual_gbm`, `residual_gp` (physics forecast + learned error) |
| Universal differential equation | `ude`: physics law times `exp(g(z))`, `g` a small network. Starts as pure physics and cannot predict a gain |
| Sequence | `gru`, `transformer` (same form with memory of usage); `gru_free` drops the physics as a comparison |
| PINN | `pinn`: physics residual and monotonicity by autodiff, at collocation points beyond the training range |
| Raw signal | `curve_cnn`: 1-D CNN on one session's dQ/dV curve |
| Nowcast | `ridge`, `gbm_monotone`, `gp`, `physics_reading`, `residual_gbm` |

With so few packs, expect the simplest physics-based models to be hard to beat. Whether a neural term earns its place is exactly what the
comparison measures.

### Ablation ladder

| Run | Features | Reading |
|---|---|---|
| `run_A` | everything, including BMS outputs | contaminated, an upper bound only |
| `run_B` | physics + curve + usage history | BMS-free, but has time proxies |
| `run_C` | physics + curve | BMS-free, no time proxies |
| `run_D` | resistance, temperature, load, cell spread | strictest: no capacity-bearing feature at all |

### Leakage rules (each one has a test in `tests/test_soh.py`)

1. Models with features from a session are trained and scored on the leave-one-out label.
2. Training labels are recomputed from data before each cutoff, because a smoothed label looks into the future.
3. Forecasters never receive a future voltage: it would already contain that session's capacity. Only plannable usage is allowed.
4. Reference curves and hyperparameters come from the fit fleet.
5. The held-out fleet is scored once. `test-final` writes a lock file and refuses to repeat without `--force-rescore`, which is recorded.

## Running it

Needs `data/processed/analysis_ready_timeseries.parquet` and `flights_with_qc.csv`, from `python -m eplanes all`.

```bash
pip install -r requirements-soh.txt
python -m eplanes.soh check        # confirms the inputs
python -m eplanes.soh all          # extract, label, validate, train, report
# after the results have been reviewed, and only then:
python -m eplanes.soh test-final
python -m eplanes.soh report
```

Each stage can be run alone (`extract`, `label`, `validate`, `train`, `report`). `extract` is cached per session, so the 10 Hz table is read
once. `--quick` trains for a few epochs to confirm things run. `--models persistence physics_bayes ude` limits the models.
`python -m eplanes.soh selftest` runs everything on simulated data.

All settings are in `config/soh.yaml`. Everything lands in `outputs/soh/`, and `report` packs it into `outputs/soh_results_<time>.zip`: tables,
figures and logs, no raw telemetry.

## The simulator

`eplanes/soh/simulate.py` generates sessions from an equivalent-circuit pack model with **known true capacity** and a BMS that behaves badly
on purpose: revisions during charges, large false episodes, and SOC scaled by the BMS's own capacity estimate. It writes the real file formats.
It exists to prove the method can work, and to catch bugs. It says nothing about the real fleet.

## Limits

* **The label assumes the voltage curve keeps its shape with age.** A shift along the curve is harmless, because `c_k` absorbs it. A stretch is
  not: it is read as a capacity change. V6 catches an uneven change but an even stretch passes it. Voltage alone cannot rule this out; a single
  reference capacity test on one aircraft can, and would also calibrate the label.
* **SOH here is relative to each pack's early life in this dataset**, not to the factory value.
* **Only logged sessions count.** Unlogged usage is missing from the Ah totals.
* **Few packs.** Two fleets, effectively three aircraft. Conclusions about models are weak, and calendar and cycling fade cannot be
  separated well within one aircraft, because they rise together.
* **Conditional forecasts.** Forecasts use the usage that really followed, so they answer "if the aircraft is used like this".
