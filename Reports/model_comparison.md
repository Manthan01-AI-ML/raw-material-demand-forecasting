# Model Comparison Report
Generated: 2025-08-25 12:55

| Model          |     RMSE |      MAPE |
|:---------------|---------:|----------:|
| RandomForest   |  5.68638 | 0.0401964 |
| XGBoost        |  5.6383  | 0.0401135 |
| Hybrid(RF+XGB) |  5.58865 | 0.0399054 |
| SARIMA         | 26.2186  | 0.214741  |

### Interpretation
- Hybrid performs best with ~4% MAPE.
- RF/XGB are solid at ~4% MAPE too.
- SARIMA lags behind (~21% MAPE).

### Business Impact
Moving from 21% error to 4% error = safer procurement planning.
