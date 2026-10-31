# HAPS Link-Quality Prediction Using a Documented Channel Model

## Technical Report Draft — PS-018

Team Strato · A3-PS018-TC381 · 31 October 2026

> No official PS-018 technical-report template was present in the supplied repository or structure/presentation PDFs. This is a content-complete Markdown draft and must be transferred to any organizer-provided format if one becomes available.

## 1. Objective

The project demonstrates a reproducible HAPS-to-ground link-quality workflow. It calculates geometry and propagation loss, converts loss into received power and SNR, compares clear-air and rain-affected scenarios, and evaluates whether a Random Forest can approximate the synthetic analytical reference on held-out trajectories.

The results are simulation results, not measurements from a deployed HAPS system.

## 2. Configuration and assumptions

| Quantity | Value |
|---|---:|
| HAPS altitude | 20 km |
| Receiver height | 1.5 m |
| Frequency | 2 GHz |
| Transmit power | 30 dBm |
| Transmit antenna gain | 20 dBi |
| Receive antenna gain | 5 dBi |
| Bandwidth | 10 MHz |
| Noise figure | 5 dB |
| Thermal noise density | −174 dBm/Hz |
| Earth radius | 6371 km |
| Rain rate | 25 mm/h |
| Rain-layer height | 5 km |
| Polarization | Vertical |
| P.528 time percentage | 50% |

Receiver trajectories use ground distances from 10 to 100 km. Eight trajectories are generated with different starts and step sizes, producing 470 samples across two scenarios.

## 3. Methodology

### 3.1 Geometry

For ground arc distance (d), Earth radius (R), and HAPS altitude (h), the implementation uses a spherical-Earth geometry model. The central angle is:

\[
\phi = d/R
\]

The slant range is:

\[
\rho = \sqrt{h^2 + 2R(R+h)(1-\cos\phi)}
\]

Elevation is calculated from the corresponding vertical and horizontal components with `atan2`.

At the 50 km reference point, the verified geometry is 53.92435 km slant range and 21.54551° elevation.

### 3.2 Propagation

Basic transmission loss is calculated by the pinned Py528 5.0.1 implementation of ITU-R P.528-5 using the project’s fixed frequency, terminal heights, vertical polarization, 50% time percentage, and reflection setting. The implementation is in `src/propagation.py`; the model is not replaced by a free-space approximation.

At 50 km, the verified basic path loss is 133.0669946 dB.

### 3.3 Link budget

Received signal power is:

\[
P_r = P_t + G_t + G_r - L_{total}
\]

Receiver noise power is:

\[
N = -174 + 10\log_{10}(B) + NF
\]

SNR is:

\[
SNR = P_r - N
\]

The dashboard calls the parameterized function in `src/link_budget.py`, so UI controls are not reimplemented as separate equations.

### 3.4 Environmental scenarios

Clear air uses the P.528 basic path loss only. Rain affected adds the existing ITU-R P.838-specific attenuation helper. The project approximates an effective rain path as:

\[
L_{rain} = \gamma_R \min(h_{rain}/\sin(elevation), 10\text{ km})
\]

This is a simplified scenario-comparison approximation, not a complete atmospheric rain-path or gaseous-attenuation model.

## 4. Dataset and ML evaluation

The dataset contains 470 rows: 235 clear-air and 235 rain-affected samples across eight trajectory IDs. Features are ground distance, slant range, and elevation angle. Received power in dBm is the primary ML target required by the supplied PS-018 material; SNR in dB is evaluated as a secondary target. Neither target is used as an input feature.

For each scenario, `LeaveOneGroupOut` holds out one complete trajectory at a time. A 100-tree Random Forest with random state 42 is trained on the remaining trajectories. Predictions are saved in `data/prediction_results.csv`, and baseline comparisons are saved in `data/baseline_comparison.csv`.

The exact checked-in MAE values are:

| Scenario | Samples | Received-power MAE | SNR MAE |
|---|---:|---:|---:|
| Clear air | 235 | 0.0134709 dB | 0.0137753 dB |
| Rain affected | 235 | 0.0130351 dB | 0.0136983 dB |

These values measure approximation to synthetic analytical labels in matching
physical quantities. They do not establish real-world prediction accuracy.
Because trajectories cover overlapping ground-distance ranges, held-out
evaluation is mainly interpolation within the same simulated geometry domain.

Prediction rows are saved with explicit columns including
`predicted_received_power_dbm` and `predicted_snr_db`.

### 4.1 Computational-cost measurement

`src/inference_benchmark.py` fits the same held-out models before timing, then
uses repeated `time.perf_counter()` measurements around only
`RandomForestRegressor.predict` calls. It reports total and per-sample time,
sample count, repetitions, Python/platform/scikit-learn versions, and explicit
flags that model loading and training are excluded. The checked-in result is
`data/inference_benchmark.csv`; values are reproducible in method and structure
but depend on machine load and hardware.

## 5. Sensitivity analysis

The verified sensitivity analysis obtains its reference path loss directly from P.528 at 50 km in clear air. It varies five link-budget parameters—transmit power, transmit gain, receive gain, bandwidth, and noise figure—and separately varies propagation assumptions through ground distance and environmental scenario. Results include units and provenance in `data/sensitivity_results.csv`; the plot is `data/sensitivity_analysis.png`.

At the reference point:

| Scenario | Basic path loss | Rain attenuation | Total path loss | SNR |
|---|---:|---:|---:|---:|
| Clear air | 133.0669946 dB | 0 dB | 133.0669946 dB | 20.9330054 dB |
| Rain affected | 133.0669946 dB | 0.0214102 dB | 133.0884047 dB | 20.9115953 dB |

## 6. Demonstration

The Streamlit dashboard in `app.py` exposes ground distance, transmit power, antenna gains, receiver noise figure, bandwidth, and scenario controls. It displays geometry, total path loss, attenuation, received power, and SNR. It plots baseline scenario curves, a current-control analytical marker, precomputed ML comparisons, and the verified sensitivity outputs.

The ML panel is deliberately labeled as precomputed. It does not retrain a model live and does not recompute ML predictions for changed link-budget controls.

Launch from the repository root:

```bash
source .venv/bin/activate
streamlit run app.py
```

## 7. Verification

The Phase 5 verification run includes the complete automated suite, targeted
dashboard checks, `python -m pip check`, Streamlit import/AppTest rendering,
and a real local server launch. The final counts are reported with the
submission handoff after the run.

## 8. Limitations and conclusion

The demonstration is technically reproducible within its documented synthetic assumptions. Its main limitations are the simplified rain model, fixed frequency and atmospheric assumptions, synthetic target labels, overlapping trajectory domains, and lack of real-world measurement validation. The supplied architecture material mentions P.525/P.676/P.835, whereas the current verified implementation uses P.528-5 and P.838; that model-stack scope should be confirmed before submission. The result supports comparison and education about the modeled link budget; it should not be presented as field-validated HAPS performance or independently validated ML generalization.

## 9. References

- ITU-R P.528-5, aeronautical propagation prediction method, through the pinned Py528 implementation.
- ITU-R P.838 rain-specific attenuation model, through `itur`.
- Supplied PS-018 structure/presentation material: `Proposed-structure-hackathon.pdf` and `AIORI3_Team_Presentation.pdf`.
