# Final Report: Causal Discovery and Counterfactual Simulation in Multivariate Forecasting

IIT Industrial Training Final Project - AIML.

This report consolidates the multi-unit CMAPSS experiments produced by `scripts/run_full_cmapss_experiment.py` and `scripts/run_all_cmapss_experiments.py`.

## Cross-Dataset Comparison

```
dataset                model     period      mae     rmse       mape      smape
  FD001               ARIMAX post-shock 0.377690 0.465352 193.810107  76.420304
  FD001        Causal Engine post-shock 0.430759 0.517924 237.855411  84.200973
  FD001 Persistence Baseline post-shock 0.568112 0.659846 313.118361  89.255640
  FD002               ARIMAX post-shock 0.082697 0.097154   9.994965   9.712389
  FD002        Causal Engine post-shock 1.126165 1.407325 143.826273 116.335438
  FD002 Persistence Baseline post-shock 1.129801 1.411487 144.271333 116.459864
  FD003        Causal Engine post-shock 0.375087 0.459085 251.929284  84.301362
  FD003 Persistence Baseline post-shock 0.434553 0.516006 299.747072  90.013484
  FD003               ARIMAX post-shock 0.568499 0.669828 250.837019 128.459136
  FD004               ARIMAX post-shock 0.080170 0.094190   9.993192   9.710952
  FD004        Causal Engine post-shock 1.062766 1.336732 135.690812 115.437127
  FD004 Persistence Baseline post-shock 1.069662 1.347933 137.182034 115.123093
```

Headline metrics: MAE and RMSE on the held-out post-shock window. MAPE is reported but is unstable on standardized targets that pass through zero.

## Shock Scenarios

See `docs/SHOCK_SCENARIOS.md` for the full catalogue and the domain reasoning behind each scenario. The headline results above use scenario S1 (step shock on `s11`).

## Per-Dataset Results

## FD001

- Best model on headline scenario: **ARIMAX**
- Best MAE: `0.3777` | Best RMSE: `0.4654` | sMAPE: `76.42%`

### Headline Forecast Metrics (mean over all held-out engines, S1)

```
               model     period      mae     rmse       mape     smape
              ARIMAX post-shock 0.377690 0.465352 193.810107 76.420304
       Causal Engine post-shock 0.430759 0.517924 237.855411 84.200973
Persistence Baseline post-shock 0.568112 0.659846 313.118361 89.255640
```

### All Shock Scenarios

```
          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.377690 0.063231   0.465352  0.072398 193.810107   76.420304  42.515381       30
   S1_step_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
   S1_step_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
S2_gradual_s11_pos               ARIMAX post-shock  0.378337 0.063422   0.466377  0.072441 195.442476   76.404589  42.413671       30
S2_gradual_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
S2_gradual_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
  S3_pulse_s11_pos               ARIMAX post-shock  0.375178 0.062713   0.462587  0.072383 188.292897   76.365346  42.525565       30
  S3_pulse_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
  S3_pulse_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
    S4_step_s7_neg               ARIMAX post-shock  0.375811 0.064026   0.464053  0.074434 185.631837   76.579740  42.530745       30
    S4_step_s7_neg        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
    S4_step_s7_neg Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
    S5_step_s2_pos               ARIMAX post-shock  0.376248 0.063268   0.464134  0.073138 189.055478   76.385001  42.500334       30
    S5_step_s2_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
    S5_step_s2_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
```

### Forecast Graph Used

```
cause effect  confidence
  s11    s12       1.000
  s11    s15       1.000
  s11     s2       1.000
  s11    s20       1.000
  s11     s3       1.000
  s11     s4       1.000
  s11     s7       1.000
  s12     s2       1.000
  s12    s20       1.000
  s12     s4       1.000
  s12     s7       1.000
  s12    s15       0.900
  s12     s3       0.625
```

### Top Edge Stability

```
cause effect  bootstrap_frequency confidence  used_in_simulation                      method
  s11    s12                  1.0       HIGH                True lag1_predictive_correlation
  s11    s15                  1.0       HIGH                True lag1_predictive_correlation
  s11     s2                  1.0       HIGH                True lag1_predictive_correlation
  s11    s20                  1.0       HIGH                True lag1_predictive_correlation
  s11     s3                  1.0       HIGH                True lag1_predictive_correlation
  s11     s4                  1.0       HIGH                True lag1_predictive_correlation
  s11     s7                  1.0       HIGH                True lag1_predictive_correlation
  s12    s11                  1.0       HIGH                True lag1_predictive_correlation
  s12     s2                  1.0       HIGH                True lag1_predictive_correlation
  s12    s20                  1.0       HIGH                True lag1_predictive_correlation
```

### Run Summary

# FD001 Multi-unit Experiment Summary

Train units: `70` engines (`1` to `70`)

Held-out units: `30` engines (`71` to `100`)

Variables: `s2, s3, s4, s7, s11, s12, s15, s20`

Differenced variables: `none`

Discovery sample rows: `400`

Bootstrap runs: `80`

Max parents per target: `2`

Temporal edge threshold: `0.50`

PC/FCI agreement rate: `0.992`

Cohen's Kappa: `0.982`

Forecast graph source: `temporal_predictive_bootstrap`

Headline scenario: `S1_step_s11_pos`

## Temporal Edges Used

cause effect  confidence
  s11    s12       1.000
  s11    s15       1.000
  s11     s2       1.000
  s11    s20       1.000
  s11     s3       1.000
  s11     s4       1.000
  s11     s7       1.000
  s12     s2       1.000
  s12    s20       1.000
  s12     s4       1.000
  s12     s7       1.000
  s12    s15       0.900
  s12     s3       0.625

## Headline Forecast Metrics (mean over all held-out engines, scenario S1_step_s11_pos)

               model     period      mae     rmse       mape     smape
              ARIMAX post-shock 0.377690 0.465352 193.810107 76.420304
       Causal Engine post-shock 0.430759 0.517924 237.855411 84.200973
Persistence Baseline post-shock 0.568112 0.659846 313.118361 89.255640

## All Scenarios (mean +/- std over held-out engines)

          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.377690 0.063231   0.465352  0.072398 193.810107   76.420304  42.515381       30
   S1_step_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
   S1_step_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
S2_gradual_s11_pos               ARIMAX post-shock  0.378337 0.063422   0.466377  0.072441 195.442476   76.404589  42.413671       30
S2_gradual_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
S2_gradual_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
  S3_pulse_s11_pos               ARIMAX post-shock  0.375178 0.062713   0.462587  0.072383 188.292897   76.365346  42.525565       30
  S3_pulse_s11_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
  S3_pulse_s11_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
    S4_step_s7_neg               ARIMAX post-shock  0.375811 0.064026   0.464053  0.074434 185.631837   76.579740  42.530745       30
    S4_step_s7_neg        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
    S4_step_s7_neg Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30
    S5_step_s2_pos               ARIMAX post-shock  0.376248 0.063268   0.464134  0.073138 189.055478   76.385001  42.500334       30
    S5_step_s2_pos        Causal Engine post-shock  0.430759 0.126039   0.517924  0.141355 237.855411   84.200973  52.638692       30
    S5_step_s2_pos Persistence Baseline post-shock  0.568112 0.330849   0.659846  0.325267 313.118361   89.255640  45.310029       30


## FD002

- Best model on headline scenario: **ARIMAX**
- Best MAE: `0.0827` | Best RMSE: `0.0972` | sMAPE: `9.71%`

### Headline Forecast Metrics (mean over all held-out engines, S1)

```
               model     period      mae     rmse       mape      smape
              ARIMAX post-shock 0.082697 0.097154   9.994965   9.712389
       Causal Engine post-shock 1.126165 1.407325 143.826273 116.335438
Persistence Baseline post-shock 1.129801 1.411487 144.271333 116.459864
```

### All Shock Scenarios

```
          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.082697 0.012937   0.097154  0.013021   9.994965    9.712389   1.494879       78
   S1_step_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
   S1_step_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
S2_gradual_s11_pos               ARIMAX post-shock  0.087989 0.014336   0.107845  0.015300  10.433106   10.047958   1.581309       78
S2_gradual_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
S2_gradual_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
  S3_pulse_s11_pos               ARIMAX post-shock  0.067883 0.012661   0.087350  0.017632   8.888190    8.901054   1.584245       78
  S3_pulse_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
  S3_pulse_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
    S4_step_s7_neg               ARIMAX post-shock  0.059670 0.009728   0.071431  0.010735   8.181407    8.266842   1.594508       78
    S4_step_s7_neg        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
    S4_step_s7_neg Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
    S5_step_s2_pos               ARIMAX post-shock  0.058966 0.012567   0.075747  0.014118   8.051051    8.533564   2.111476       78
    S5_step_s2_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
    S5_step_s2_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
```

### Forecast Graph Used

```
cause effect  confidence
   s2     s3         1.0
   s3     s4         1.0
  s11    s15         1.0
```

### Top Edge Stability

```
Empty DataFrame
Columns: [cause, effect, bootstrap_frequency, confidence, used_in_simulation, method]
Index: []
```

### Run Summary

# FD002 Multi-unit Experiment Summary

Train units: `182` engines (`1` to `182`)

Held-out units: `78` engines (`183` to `260`)

Variables: `s2, s3, s4, s7, s11, s12, s15, s20`

Differenced variables: `none`

Discovery sample rows: `400`

Bootstrap runs: `80`

Max parents per target: `2`

Temporal edge threshold: `0.50`

PC/FCI agreement rate: `1.000`

Cohen's Kappa: `1.000`

Forecast graph source: `domain_prior_temporal_fallback`

Headline scenario: `S1_step_s11_pos`

## Temporal Edges Used

cause effect  confidence
   s2     s3         1.0
   s3     s4         1.0
  s11    s15         1.0

## Headline Forecast Metrics (mean over all held-out engines, scenario S1_step_s11_pos)

               model     period      mae     rmse       mape      smape
              ARIMAX post-shock 0.082697 0.097154   9.994965   9.712389
       Causal Engine post-shock 1.126165 1.407325 143.826273 116.335438
Persistence Baseline post-shock 1.129801 1.411487 144.271333 116.459864

## All Scenarios (mean +/- std over held-out engines)

          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.082697 0.012937   0.097154  0.013021   9.994965    9.712389   1.494879       78
   S1_step_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
   S1_step_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
S2_gradual_s11_pos               ARIMAX post-shock  0.087989 0.014336   0.107845  0.015300  10.433106   10.047958   1.581309       78
S2_gradual_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
S2_gradual_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
  S3_pulse_s11_pos               ARIMAX post-shock  0.067883 0.012661   0.087350  0.017632   8.888190    8.901054   1.584245       78
  S3_pulse_s11_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
  S3_pulse_s11_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
    S4_step_s7_neg               ARIMAX post-shock  0.059670 0.009728   0.071431  0.010735   8.181407    8.266842   1.594508       78
    S4_step_s7_neg        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
    S4_step_s7_neg Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78
    S5_step_s2_pos               ARIMAX post-shock  0.058966 0.012567   0.075747  0.014118   8.051051    8.533564   2.111476       78
    S5_step_s2_pos        Causal Engine post-shock  1.126165 0.363925   1.407325  0.359172 143.826273  116.335438  22.010485       78
    S5_step_s2_pos Persistence Baseline post-shock  1.129801 0.358847   1.411487  0.351509 144.271333  116.459864  22.276186       78


## FD003

- Best model on headline scenario: **Causal Engine**
- Best MAE: `0.3751` | Best RMSE: `0.4591` | sMAPE: `84.30%`

### Headline Forecast Metrics (mean over all held-out engines, S1)

```
               model     period      mae     rmse       mape      smape
       Causal Engine post-shock 0.375087 0.459085 251.929284  84.301362
Persistence Baseline post-shock 0.434553 0.516006 299.747072  90.013484
              ARIMAX post-shock 0.568499 0.669828 250.837019 128.459136
```

### All Shock Scenarios

```
          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
   S1_step_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
   S1_step_s11_pos               ARIMAX post-shock  0.568499 0.117734   0.669828  0.117047 250.837019  128.459136  38.443710       30
S2_gradual_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
S2_gradual_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
S2_gradual_s11_pos               ARIMAX post-shock  0.568069 0.116210   0.669564  0.116078 253.046854  128.272517  38.358656       30
  S3_pulse_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
  S3_pulse_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
  S3_pulse_s11_pos               ARIMAX post-shock  0.576303 0.123087   0.677296  0.121138 251.113167  130.245839  38.168625       30
    S4_step_s7_neg        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
    S4_step_s7_neg Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
    S4_step_s7_neg               ARIMAX post-shock  0.582621 0.125536   0.683657  0.124001 251.600838  131.973350  37.962352       30
    S5_step_s2_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
    S5_step_s2_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
    S5_step_s2_pos               ARIMAX post-shock  0.578444 0.123609   0.679793  0.122307 252.282348  130.820984  38.117197       30
```

### Forecast Graph Used

```
cause effect  confidence
  s11     s2         1.0
  s11     s3         1.0
  s11     s4         1.0
  s12     s7         1.0
  s15    s12         1.0
  s15    s20         1.0
  s15     s7         1.0
   s2     s4         1.0
   s4     s3         1.0
   s7    s20         1.0
```

### Top Edge Stability

```
cause effect  bootstrap_frequency confidence  used_in_simulation                      method
  s11     s2                  1.0       HIGH                True lag1_predictive_correlation
  s11     s3                  1.0       HIGH                True lag1_predictive_correlation
  s11     s4                  1.0       HIGH                True lag1_predictive_correlation
  s12     s7                  1.0       HIGH                True lag1_predictive_correlation
  s15    s12                  1.0       HIGH                True lag1_predictive_correlation
  s15    s20                  1.0       HIGH                True lag1_predictive_correlation
  s15     s7                  1.0       HIGH                True lag1_predictive_correlation
   s2     s4                  1.0       HIGH                True lag1_predictive_correlation
  s20    s15                  1.0       HIGH                True lag1_predictive_correlation
   s4    s11                  1.0       HIGH                True lag1_predictive_correlation
```

### Run Summary

# FD003 Multi-unit Experiment Summary

Train units: `70` engines (`1` to `70`)

Held-out units: `30` engines (`71` to `100`)

Variables: `s2, s3, s4, s7, s11, s12, s15, s20`

Differenced variables: `none`

Discovery sample rows: `400`

Bootstrap runs: `80`

Max parents per target: `2`

Temporal edge threshold: `0.50`

PC/FCI agreement rate: `0.983`

Cohen's Kappa: `0.955`

Forecast graph source: `temporal_predictive_bootstrap`

Headline scenario: `S1_step_s11_pos`

## Temporal Edges Used

cause effect  confidence
  s11     s2         1.0
  s11     s3         1.0
  s11     s4         1.0
  s12     s7         1.0
  s15    s12         1.0
  s15    s20         1.0
  s15     s7         1.0
   s2     s4         1.0
   s4     s3         1.0
   s7    s20         1.0

## Headline Forecast Metrics (mean over all held-out engines, scenario S1_step_s11_pos)

               model     period      mae     rmse       mape      smape
       Causal Engine post-shock 0.375087 0.459085 251.929284  84.301362
Persistence Baseline post-shock 0.434553 0.516006 299.747072  90.013484
              ARIMAX post-shock 0.568499 0.669828 250.837019 128.459136

## All Scenarios (mean +/- std over held-out engines)

          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
   S1_step_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
   S1_step_s11_pos               ARIMAX post-shock  0.568499 0.117734   0.669828  0.117047 250.837019  128.459136  38.443710       30
S2_gradual_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
S2_gradual_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
S2_gradual_s11_pos               ARIMAX post-shock  0.568069 0.116210   0.669564  0.116078 253.046854  128.272517  38.358656       30
  S3_pulse_s11_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
  S3_pulse_s11_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
  S3_pulse_s11_pos               ARIMAX post-shock  0.576303 0.123087   0.677296  0.121138 251.113167  130.245839  38.168625       30
    S4_step_s7_neg        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
    S4_step_s7_neg Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
    S4_step_s7_neg               ARIMAX post-shock  0.582621 0.125536   0.683657  0.124001 251.600838  131.973350  37.962352       30
    S5_step_s2_pos        Causal Engine post-shock  0.375087 0.085956   0.459085  0.093651 251.929284   84.301362  43.597702       30
    S5_step_s2_pos Persistence Baseline post-shock  0.434553 0.126808   0.516006  0.125122 299.747072   90.013484  44.827137       30
    S5_step_s2_pos               ARIMAX post-shock  0.578444 0.123609   0.679793  0.122307 252.282348  130.820984  38.117197       30


## FD004

- Best model on headline scenario: **ARIMAX**
- Best MAE: `0.0802` | Best RMSE: `0.0942` | sMAPE: `9.71%`

### Headline Forecast Metrics (mean over all held-out engines, S1)

```
               model     period      mae     rmse       mape      smape
              ARIMAX post-shock 0.080170 0.094190   9.993192   9.710952
       Causal Engine post-shock 1.062766 1.336732 135.690812 115.437127
Persistence Baseline post-shock 1.069662 1.347933 137.182034 115.123093
```

### All Shock Scenarios

```
          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.080170 0.011466   0.094190  0.011270   9.993192    9.710952   1.660493       75
   S1_step_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
   S1_step_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
S2_gradual_s11_pos               ARIMAX post-shock  0.086146 0.013459   0.104071  0.013529  10.593623   10.209323   1.791346       75
S2_gradual_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
S2_gradual_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
  S3_pulse_s11_pos               ARIMAX post-shock  0.065344 0.011742   0.084881  0.016249   8.751451    8.760701   1.627489       75
  S3_pulse_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
  S3_pulse_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
    S4_step_s7_neg               ARIMAX post-shock  0.059923 0.008635   0.072176  0.009400   8.363496    8.404729   1.597096       75
    S4_step_s7_neg        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
    S4_step_s7_neg Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
    S5_step_s2_pos               ARIMAX post-shock  0.059398 0.010887   0.076377  0.011943   8.176169    8.649566   1.848681       75
    S5_step_s2_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
    S5_step_s2_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
```

### Forecast Graph Used

```
cause effect  confidence
   s2     s3         1.0
   s3     s4         1.0
  s11    s15         1.0
```

### Top Edge Stability

```
Empty DataFrame
Columns: [cause, effect, bootstrap_frequency, confidence, used_in_simulation, method]
Index: []
```

### Run Summary

# FD004 Multi-unit Experiment Summary

Train units: `174` engines (`1` to `174`)

Held-out units: `75` engines (`175` to `249`)

Variables: `s2, s3, s4, s7, s11, s12, s15, s20`

Differenced variables: `none`

Discovery sample rows: `400`

Bootstrap runs: `80`

Max parents per target: `2`

Temporal edge threshold: `0.50`

PC/FCI agreement rate: `0.992`

Cohen's Kappa: `0.967`

Forecast graph source: `domain_prior_temporal_fallback`

Headline scenario: `S1_step_s11_pos`

## Temporal Edges Used

cause effect  confidence
   s2     s3         1.0
   s3     s4         1.0
  s11    s15         1.0

## Headline Forecast Metrics (mean over all held-out engines, scenario S1_step_s11_pos)

               model     period      mae     rmse       mape      smape
              ARIMAX post-shock 0.080170 0.094190   9.993192   9.710952
       Causal Engine post-shock 1.062766 1.336732 135.690812 115.437127
Persistence Baseline post-shock 1.069662 1.347933 137.182034 115.123093

## All Scenarios (mean +/- std over held-out engines)

          scenario                model     period  mae_mean  mae_std  rmse_mean  rmse_std  mape_mean  smape_mean  smape_std  n_units
   S1_step_s11_pos               ARIMAX post-shock  0.080170 0.011466   0.094190  0.011270   9.993192    9.710952   1.660493       75
   S1_step_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
   S1_step_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
S2_gradual_s11_pos               ARIMAX post-shock  0.086146 0.013459   0.104071  0.013529  10.593623   10.209323   1.791346       75
S2_gradual_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
S2_gradual_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
  S3_pulse_s11_pos               ARIMAX post-shock  0.065344 0.011742   0.084881  0.016249   8.751451    8.760701   1.627489       75
  S3_pulse_s11_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
  S3_pulse_s11_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
    S4_step_s7_neg               ARIMAX post-shock  0.059923 0.008635   0.072176  0.009400   8.363496    8.404729   1.597096       75
    S4_step_s7_neg        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
    S4_step_s7_neg Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75
    S5_step_s2_pos               ARIMAX post-shock  0.059398 0.010887   0.076377  0.011943   8.176169    8.649566   1.848681       75
    S5_step_s2_pos        Causal Engine post-shock  1.062766 0.349448   1.336732  0.348577 135.690812  115.437127  22.221900       75
    S5_step_s2_pos Persistence Baseline post-shock  1.069662 0.349706   1.347933  0.344384 137.182034  115.123093  22.620021       75


## Generated Figures

- `reports/figures/causal_dag_fd001_unit1.html`
- `reports/figures/causal_dag_multiunit_fd001.html`
- `reports/figures/causal_dag_multiunit_fd002.html`
- `reports/figures/causal_dag_multiunit_fd003.html`
- `reports/figures/causal_dag_multiunit_fd004.html`
- `reports/figures/counterfactual_multiunit_fd001.html`
- `reports/figures/counterfactual_multiunit_fd002.html`
- `reports/figures/counterfactual_multiunit_fd003.html`
- `reports/figures/counterfactual_multiunit_fd004.html`
- `reports/figures/counterfactual_s11_to_s4.html`
- `reports/figures/model_comparison_fd001_unit1.html`
- `reports/figures/model_comparison_multiunit_fd001.html`
- `reports/figures/model_comparison_multiunit_fd002.html`
- `reports/figures/model_comparison_multiunit_fd003.html`
- `reports/figures/model_comparison_multiunit_fd004.html`

## Reproduction

```
python scripts/run_all_cmapss_experiments.py
python scripts/generate_final_report.py
streamlit run apps/streamlit_dashboard.py
```
