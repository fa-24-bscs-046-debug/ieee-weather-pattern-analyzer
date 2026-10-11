"""IEEE LGU AI/ML Cohort One - Week 3: Weather Pattern Analyzer (NumPy & Pandas)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EXTREME_TEMP_C = 35.0      # hot day threshold
EXTREME_RAIN_MM = 10.0     # heavy rain threshold
VALID_TEMP = (-10.0, 55.0)  # readings outside are treated as sensor errors


def load_and_clean(path):
    """Load daily weather CSV, fix dates, remove bad rows and fill gaps."""
    df = pd.read_csv(path)
    log = {"Raw rows": len(df)}

    # Dates: invalid ones become NaT and are dropped (cannot be placed in time)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log["Rows with invalid dates dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # Duplicate dates: keep first reading
    before = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    log["Duplicate dates removed"] = before - len(df)

    # Text such as "abc" -> NaN
    for col in ["temperature_c", "humidity_pct", "rainfall_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physically impossible values -> NaN (sensor errors)
    bad_temp = ~df["temperature_c"].between(*VALID_TEMP) & df["temperature_c"].notna()
    df.loc[bad_temp, "temperature_c"] = np.nan
    log["Temperature outliers set to missing"] = int(bad_temp.sum())
    df.loc[~df["humidity_pct"].between(0, 100), "humidity_pct"] = np.nan
    df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    for col in ["temperature_c", "humidity_pct", "rainfall_mm"]:
        log[f"Missing/invalid {col}"] = int(df[col].isna().sum())

    # Fill strategy: temperature changes smoothly -> interpolate;
    # humidity -> column mean; rainfall unknown -> 0 mm (no false rain)
    df["temperature_c"] = df["temperature_c"].interpolate(limit_direction="both")
    df["humidity_pct"] = df["humidity_pct"].fillna(df["humidity_pct"].mean())
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)
    log["Clean rows"] = len(df)
    return df, log


def analyze(df):
    """NumPy statistics, monthly grouping and extreme-day detection."""
    temp = df["temperature_c"].to_numpy()
    hum = df["humidity_pct"].to_numpy()
    rain = df["rainfall_mm"].to_numpy()

    extreme = (temp > EXTREME_TEMP_C) | (rain > EXTREME_RAIN_MM)
    df["is_extreme"] = extreme
    df["day_to_day_change"] = np.append(0.0, np.diff(temp))
    df["month"] = df["date"].dt.to_period("M").astype(str)

    overall = {
        "avg_temp": np.mean(temp),
        "avg_humidity": np.mean(hum),
        "temp_range": np.max(temp) - np.min(temp),
        "max_daily_swing": np.max(np.abs(np.diff(temp))),
        "hot_days": int(np.count_nonzero(temp > EXTREME_TEMP_C)),
        "rainy_days": int(np.count_nonzero(rain > EXTREME_RAIN_MM)),
        "extreme_days": int(np.count_nonzero(extreme)),
    }

    monthly = df.groupby("month").agg(
        avg_temp=("temperature_c", "mean"),
        max_temp=("temperature_c", "max"),
        min_temp=("temperature_c", "min"),
        avg_humidity=("humidity_pct", "mean"),
        total_rain=("rainfall_mm", "sum"),
        extreme_days=("is_extreme", "sum"),
    ).reset_index()
    monthly["temp_range"] = monthly["max_temp"] - monthly["min_temp"]
    return overall, monthly


def compare_with_targets(monthly, targets_path):
    """Merge monthly summary with historical targets on the shared 'month' key."""
    targets = pd.read_csv(targets_path)
    merged = monthly.merge(targets, on="month", how="left")
    merged["temp_diff"] = merged["avg_temp"] - merged["historical_avg_temp"]
    merged["rain_pct_of_target"] = merged["total_rain"] / merged["target_rainfall_mm"] * 100
    return merged


def print_report(overall, merged, extreme_days_df):
    line = "=" * 62
    print(f"\n{line}\n{'WEATHER PATTERN ANALYSIS REPORT':^62}\n{line}")
    print(f"Average temperature : {overall['avg_temp']:.2f} C")
    print(f"Average humidity    : {overall['avg_humidity']:.2f} %")
    print(f"Overall temp range  : {overall['temp_range']:.1f} C")
    print(f"Max day-to-day swing: {overall['max_daily_swing']:.1f} C")
    print(f"Extreme days        : {overall['extreme_days']} "
          f"(>{EXTREME_TEMP_C:.0f}C: {overall['hot_days']}, >{EXTREME_RAIN_MM:.0f}mm rain: {overall['rainy_days']})")

    print(f"\n{'-' * 62}\nMONTHLY SUMMARY\n{'-' * 62}")
    cols = ["month", "avg_temp", "avg_humidity", "total_rain", "temp_range", "extreme_days"]
    print(merged[cols].round(2).to_string(index=False))

    print(f"\n{'-' * 62}\nCOMPARISON WITH HISTORICAL BASELINE\n{'-' * 62}")
    for _, r in merged.iterrows():
        trend = "warmer" if r["temp_diff"] > 0 else "cooler"
        rain = "above" if r["rain_pct_of_target"] > 100 else "below"
        print(f"{r['month']}: {abs(r['temp_diff']):.2f} C {trend} than average "
              f"({r['avg_temp']:.2f} vs {r['historical_avg_temp']:.1f}); "
              f"rain {r['total_rain']:.1f} mm = {r['rain_pct_of_target']:.0f}% of target ({rain})")

    print(f"\n{'-' * 62}\nEXTREME WEATHER DAYS\n{'-' * 62}")
    show = extreme_days_df[["date", "temperature_c", "rainfall_mm"]].copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    print(show.to_string(index=False))


def main():
    df, log = load_and_clean(DATA_DIR / "daily_weather.csv")
    print("=" * 62 + f"\n{'DATA CLEANING LOG':^62}\n" + "=" * 62)
    for key, value in log.items():
        print(f"{key:<42}: {value}")
    overall, monthly = analyze(df)
    merged = compare_with_targets(monthly, DATA_DIR / "monthly_targets.csv")
    print_report(overall, merged, df[df["is_extreme"]])


if __name__ == "__main__":
    main()
