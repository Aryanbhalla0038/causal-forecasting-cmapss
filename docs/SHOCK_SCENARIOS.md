# Shock Scenarios for CMAPSS Causal Forecasting

This document defines the controlled shock scenarios used to evaluate the
Causal Forecasting Engine against the Persistence and ARIMAX baselines on
NASA CMAPSS turbofan data, along with the domain reasoning that justifies
each scenario.

## Why Shocks?

Standard regression metrics on a held-out window only test the model under
the same operating regime as training. Real engines also experience sudden
operating-point changes (throttle steps, bleed valve faults, fouling events).
A defensible causal-forecasting evaluation must therefore include
out-of-distribution shock windows where:

- the intervention variable is forced off its natural trajectory,
- the rest of the system must be predicted from causal structure,
- baselines that ignore structure (Persistence, ARIMAX without correct
  exogenous structure) are expected to degrade.

All shock scenarios below operate on standardized variables, so a magnitude
of `0.10` corresponds to roughly one tenth of one standard deviation of the
sensor's post-stationarization distribution.

## Variable Roles

| Symbol | CMAPSS Sensor | Engine Quantity | Role |
|--------|---------------|-----------------|------|
| `s2`   | T24           | LPC outlet temperature   | upstream thermal state |
| `s3`   | T30           | HPC outlet temperature   | core thermal state |
| `s4`   | T50           | LPT outlet temperature   | downstream thermal state, target |
| `s7`   | P30           | HPC outlet pressure      | core pressure |
| `s11`  | Ps30          | HPC static pressure      | intervention candidate |
| `s12`  | phi           | fuel-air ratio (derived) | combustion state |
| `s15`  | NRf bypass    | bypass-corrected speed   | flow indicator |
| `s20`  | htBleed       | bleed enthalpy           | secondary flow |

Target variable for headline metrics: `s4` (downstream LPT temperature).
Primary intervention variable: `s11` (core pressure surrogate).

## Scenario Catalogue

### S1. Step shock on `s11` (default headline scenario)

- **Type:** `step`
- **Magnitude:** `+0.10`
- **Start index:** first held-out cycle after the warm-up state
- **Duration:** to end of held-out window
- **Domain meaning:** sudden, persistent rise in HPC static pressure, e.g.
  bleed valve closure or compressor fouling causing a step in compressor
  back pressure.
- **Why it stresses baselines:** Persistence ignores the intervention
  entirely; ARIMAX treats `s11` as exogenous but cannot rebuild the
  downstream thermal cascade because it has no per-edge structural model.
- **Why it favors causal model:** the structural equation for `s4` reads
  `s11` (and chained ancestors) as inputs, so a forced shift of `s11`
  propagates through the discovered DAG to `s4`.

### S2. Gradual shock on `s11`

- **Type:** `gradual`
- **Magnitude:** `+0.15`
- **Transition steps:** `10`
- **Start index:** first held-out cycle
- **Domain meaning:** progressive HPC fouling over ~10 cycles, then a
  sustained offset, mimicking gradual degradation rather than an abrupt
  fault.
- **Use:** robustness check that the causal engine handles smooth drift, not
  just discontinuous steps.

### S3. Pulse shock on `s11`

- **Type:** `pulse`
- **Magnitude:** `+0.20`
- **Duration:** `3` cycles
- **Start index:** middle of held-out window
- **Domain meaning:** short pressure spike, e.g. transient bleed valve
  flutter, that recovers to the prior trajectory.
- **Use:** tests whether the causal engine over-reacts and whether
  recovery dynamics propagate correctly through the DAG.

### S4. Negative step on `s7`

- **Type:** `step`
- **Magnitude:** `-0.10`
- **Start index:** first held-out cycle
- **Domain meaning:** sudden drop in HPC outlet pressure consistent with a
  small surge or stall margin event.
- **Use:** confirms that the causal engine reacts symmetrically and that
  another intervention node (not just `s11`) propagates downstream to
  `s4`.

### S5. Combined upstream shock on `s2`

- **Type:** `step`
- **Magnitude:** `+0.10`
- **Domain meaning:** elevated LPC outlet temperature consistent with hot
  inlet conditions or LPC efficiency loss.
- **Use:** tests that the engine respects the upstream-to-downstream
  causal ordering by propagating an upstream temperature shift through
  `s3` -> `s4`.

## Reporting Requirements

Each scenario must report, on the same held-out window:

1. MAE, RMSE for Persistence Baseline, ARIMAX, and Causal Engine.
2. Pre-shock vs post-shock metric split where applicable (a flat persistence
   forecast looks artificially competitive on stationary windows; this is
   why post-shock is the headline split).
3. The forecast graph used (`forecast_edges_used.csv`) and its provenance
   tag from `run_summary.md` (`temporal_predictive_bootstrap`,
   `pc_temporal_edges`, or `domain_prior_temporal_fallback`).
4. Counterfactual figure for the intervention variable in `reports/figures/`.

## Implementation Pointers

- Scenario S1 is what `scripts/run_full_cmapss_experiment.py` runs by
  default.
- Scenarios S2-S5 can be reproduced by editing the `INTERVENTION`,
  `magnitude`, and `shock_type` arguments at the
  `ShockInjector.inject_dataframe_shock` call inside that script.
- All scenarios operate on the standardized representation; metrics are
  not rescaled because comparisons across models are like-for-like on the
  same target series.

## Honest Limitations

- The CMAPSS dataset does not contain ground-truth interventions, so these
  shocks are synthetic disturbances applied at evaluation time. They test
  model behavior under structured perturbations, not real fault injection.
- MAPE is unstable on standardized targets that pass through zero; MAE and
  RMSE are the headline metrics for shock evaluations.
