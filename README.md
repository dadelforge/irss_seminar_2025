# Supplementary material for the IRSS seminar 2025

This repository contains the companion materials for the IRSS technical seminar
talk. It centers on a single, self‑contained Jupyter notebook that demonstrates
an empirical approach to building mortality baselines and computing excess
mortality from weekly city‑level data.

Supplement to:

> Delforge D. 2025-11-12, Introducing an Empirical Approach to Mortality
> Baselines and Excess Mortality Calculation from Historical Records: Belgian and
> Greek City Case Studies, IRSS technical seminar [hybrid], University of Louvain,
> 1200 Brussels, Belgium

## What’s in here (succinct)

- Primary notebook: `irss_seminar.ipynb`
    - Loads Eurostat weekly all‑cause mortality for selected cities (Brussels,
      Athens, …)
    - Visualizes time series and distributions, highlighting COVID‑period
      windows
    - Implements two simple excess‑mortality approaches:
        - Percentile‑based thresholding (e.g., 95th percentile)
        - Z‑score approach using a Yeo‑Johnson power transform to approximate
          normality, then segmentation
    - Extracts and summarizes “mortality events” (duration, magnitude, excess)
      via `utils/`
- Presentation slides: `251112_irss_seminar.pdf`
- Data: `data/df_mortality_TOTAL.csv` (pre‑prepared extract for the demo)
- Utilities: lightweight helpers in `utils/` for segmentation, plotting, and
  data labels
- Plot style: `style.mplstyle` for consistent figures

## Quick start

1. Open the notebook `irss_seminar.ipynb` in Jupyter (Lab/VS Code) and run cells
   top‑to‑bottom.
2. Required Python packages (typical): `numpy`, `pandas`, `matplotlib`, `scipy`,
   `scikit-learn`, `statsmodels`, `jupyter`.
    - Example:
      `pip install numpy pandas matplotlib scipy scikit-learn statsmodels jupyter`
3. The notebook reads `data/df_mortality_TOTAL.csv` and produces figures and an
   event table.

## Data source

Eurostat — "Deaths by week, sex and NUTS 3 region" (
DOI: https://doi.org/10.2908/DEMO_R_MWK3_TS). City labels are mapped via
`utils/data.py`.

