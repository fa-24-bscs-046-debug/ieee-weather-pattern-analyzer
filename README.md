# Weather Pattern Analyzer
### IEEE LGU AI/ML Cohort One — Fall 2026
**Week 3 Assignment:** Data Analysis with NumPy & Pandas
**AI/ML Leads:** Abdullah Faisal & Alina Irshad

---

## 1. Project Overview
- Analyzes 2 months of daily weather readings (Aug–Sep 2026) from `data/daily_weather.csv`.
- Cleans sensor errors, computes NumPy statistics, groups by month.
- Merges with `data/monthly_targets.csv` to compare against historical baselines.

## 2. Key Concepts Implemented
- [x] Pandas DataFrame Loading & Cleaning (`read_csv`, `fillna`, `dropna`, `drop_duplicates`)
- [x] NumPy Statistical Calculations (Mean, Median, Standard Deviation, Sums)
- [x] Data Grouping & Aggregation (`groupby`)
- [x] Conditional Filtering & Indexing
- [x] Dataset Merging / Time-Series Resampling — monthly summary merged on `month`


## 3. How to Run the Program
```shell
pip install pandas numpy
python main.py
```


## 4. Dataset Information & Cleaning Decisions
- **Data Source:** Self-made realistic CSVs with intentional errors (no real station data). Targets file holds historical average temperature and target rainfall per month.
- **Invalid date** (`2026-13-45`): dropped, since it can't be placed in time.
- **Duplicate dates:** kept first reading only.
- **Text values** (e.g. `abc`): converted to missing with `pd.to_numeric(errors="coerce")`.
- **Temperature outlier** (99.9 °C): outside −10 to 55 °C, treated as sensor error → missing.
- **Missing temperature:** linear interpolation (temperature changes smoothly day to day).
- **Missing humidity:** filled with column mean.
- **Missing rainfall:** filled with 0 mm to avoid inventing rain.
- **Extreme day:** temperature > 35 °C **or** rainfall > 10 mm.
- **Temperature range:** dataset has one temperature per day, so range = max − min per month, plus day-to-day swing via `np.diff`.

## 5. Console Output Screenshots
**Data cleaning log**

![Cleaning Output](screenshots/cleaning_output.png)

**Summary report**

![Terminal Output](screenshots/summary_report.png)
