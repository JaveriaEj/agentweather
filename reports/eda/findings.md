# EDA findings (auto-generated from the Feature Store)

- Dataset covers 2,162 hourly records (~90 days), 2026-08-24 to 2026-11-22.
- AQI averages 72; it is worst around 16:00 and best around 05:00 local time. 14.3% of hours exceed AQI 100 (unhealthy for sensitive groups).
- Temperature ranges 18.8-37.2°C; the daily peak is typically around 13:00, average daily swing 7.3°C.
- Rain is recorded in 1.9% of hours - a strongly imbalanced target, so rain probability uses a classifier with climatology fallback.
- Temperature and humidity correlate at r=-0.55.
- AQI vs wind speed correlation r=-0.34 (negative = wind disperses pollution).
- Most extreme values (3×IQR rule) are in precip_mm (149 hours, 6.9%); they are kept (real events), not removed.

## Missing values (% of hours, after imputation)

```
                               missing_after_imputation_pct            note
aqi                                                9.158187                
temp_c                                             0.000000                
precip_mm                                          0.000000                
humidity_pct                                       0.000000                
uvi                                                0.832562                
wind_speed                                         0.000000                
pm2_5                                              9.158187                
pressure_hpa                                       0.000000                
(rows with any imputed value)                           NaN  2.78% of hours
```