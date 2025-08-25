# run_forecast.py
# Forecast next N days using trained Hybrid (RF+XGB)
# Supports per-material forecasting via one-hot dummy control.

import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

def _detect_material_dummy_cols(columns):
    """Return list of one-hot columns for Material_Name_* (may be empty if baseline-only)."""
    return [c for c in columns if c.startswith("Material_Name_")]

def _is_baseline_material(material_name, material_dummy_cols):
    """If the material has NO dedicated one-hot column, it's the baseline (all zeros)."""
    suffixes = [c.replace("Material_Name_", "") for c in material_dummy_cols]
    return material_name not in suffixes

def _apply_material_onehots(feat_dict, feature_cols, material_name):
    """Set the material one-hots correctly for the requested material."""
    mat_cols = [c for c in feature_cols if c.startswith("Material_Name_")]
    # set all zeros first
    for c in mat_cols:
        feat_dict[c] = 0.0
    # if a matching col exists, set that one to 1
    target_col = f"Material_Name_{material_name}"
    if target_col in mat_cols:
        feat_dict[target_col] = 1.0
    # if not, it's the baseline material; leaving all zeros is correct
    return feat_dict

def run_forecast(days=90, material_name=None):
    # 1) Load feature-engineered dataset used for training
    df = pd.read_csv("data/processed/feature_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 2) Load models
    rf = joblib.load("models/rf_model.pkl")
    xg = joblib.load("models/xgb_model.pkl")

    # 3) Exact training feature list (name + order)
    feature_cols = [c for c in df.columns if c not in ["Date", "Quantity_Consumed"]]

    # Identify material one-hot pattern and filter context to the requested material (if any)
    mat_cols = _detect_material_dummy_cols(feature_cols)

    if material_name:
        # Build a boolean mask to filter rows for this material in historical data:
        if _is_baseline_material(material_name, mat_cols):
            # baseline material => rows where all material dummies == 0
            mask = np.ones(len(df), dtype=bool)
            for c in mat_cols:
                mask &= (df[c] == 0)
        else:
            # one-hot exists => rows where that dummy == 1
            target_col = f"Material_Name_{material_name}"
            mask = (df[target_col] == 1)

        df_mat = df[mask].copy()
        if df_mat.empty:
            raise ValueError(f"No historical rows found for material '{material_name}'. Check name/dummies.")
        df_use = df_mat
    else:
        # No filtering => aggregate-style forecast using last row’s mix
        df_use = df

    # 4) Forecast timeline
    last_date = df_use["Date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq="D")

    # 5) Keep 30-day context with all columns we may need for carry-forward
    context_cols = set(feature_cols) | {"Date", "Quantity_Consumed"}
    last_data = df_use[[c for c in df_use.columns if c in context_cols]].iloc[-30:].copy()
    current_data = last_data.copy()

    preds = []

    for d in future_dates:
        # Start with all-zero feature row to match exact training columns
        feat = {col: 0.0 for col in feature_cols}

        # Calendar features: set both cases if present (depends on your Step 4)
        if "Month" in feature_cols: feat["Month"] = int(d.month)
        if "month" in feature_cols: feat["month"] = int(d.month)
        if "Year" in feature_cols:  feat["Year"]  = int(d.year)
        if "dayofweek" in feature_cols: feat["dayofweek"] = int(d.dayofweek)

        # Lags
        for lag in [7, 14, 30]:
            key = f"lag_{lag}"
            if key in feature_cols:
                feat[key] = float(current_data["Quantity_Consumed"].iloc[-lag])

        # Rolling means
        if "rolling_7" in feature_cols:
            feat["rolling_7"] = float(current_data["Quantity_Consumed"].iloc[-7:].mean())
        if "rolling_30" in feature_cols:
            feat["rolling_30"] = float(current_data["Quantity_Consumed"].iloc[-30:].mean())

        # Carry-forward exogenous vars if they exist in training
        last_row = current_data.iloc[-1]
        if "Lead_Time" in feature_cols and "Lead_Time" in current_data.columns:
            feat["Lead_Time"] = float(last_row["Lead_Time"])
        if "Sales_Volume" in feature_cols and "Sales_Volume" in current_data.columns:
            feat["Sales_Volume"] = float(last_row["Sales_Volume"])

        # Material one-hots: set per requested material, else copy last row's mix
        if material_name:
            feat = _apply_material_onehots(feat, feature_cols, material_name)
        else:
            for c in [c for c in feature_cols if c.startswith("Material_Name_")]:
                if c in current_data.columns:
                    feat[c] = float(last_row[c])

        # Vendor/Location dummies: copy last known (keeps latest procurement context)
        for c in [c for c in feature_cols if c.startswith("Vendor_Name_") or c.startswith("Location_")]:
            if c in current_data.columns:
                feat[c] = float(last_row[c])

        # Build X with exact column order
        X_row = pd.DataFrame([feat], columns=feature_cols)

        # Predict (Hybrid)
        rf_pred = float(rf.predict(X_row)[0])
        xg_pred = float(xg.predict(X_row)[0])
        hyb_pred = (rf_pred + xg_pred) / 2.0

        preds.append({
            "Date": d,
            "Material_Filter": material_name if material_name else "ALL",
            "Forecast": hyb_pred,
            "RF_Pred": rf_pred,
            "XGB_Pred": xg_pred
        })

        # Update rolling context for next-step lags/rolling
        upd = last_row.copy()
        upd["Date"] = d
        upd["Quantity_Consumed"] = hyb_pred
        # keep Lead_Time, Sales_Volume, dummies as carried-forward
        current_data = pd.concat([current_data, pd.DataFrame([upd])], ignore_index=True)

    forecast_df = pd.DataFrame(preds)
    os.makedirs("data/forecasts", exist_ok=True)
    suffix = material_name if material_name else "ALL"
    out_path = f"data/forecasts/forecast_{suffix}_{last_date.strftime('%Y%m%d')}.csv"
    forecast_df.to_csv(out_path, index=False)
    print("✅ Forecast saved at:", out_path)
    return forecast_df

if __name__ == "__main__":
    # Examples:
    #   python run_forecast.py        -> ALL materials context (last mix)
    #   (from Python) run_forecast(90, material_name="Cement")
    run_forecast(90)

from run_forecast import run_forecast
df_cem = run_forecast(90, material_name="Cement")

