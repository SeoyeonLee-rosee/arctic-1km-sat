# A 1-km surface air temperature reconstruction reveals terrain-driven thermal heterogeneity and ecosystem responses across Arctic land

Code for reproducing the analyses and figures in Lee et al., *"A 1-km surface air
temperature reconstruction reveals terrain-driven thermal heterogeneity and
ecosystem responses across Arctic land"* (submitted to *Nature Communications*).

This repository contains the full workflow: (1) a machine-learning model that
downscales/estimates near-surface air temperature (SAT) from MODIS satellite
products and in-situ station observations, and (2) the downstream analysis of
that SAT product together with snow-cover phenology (SCE/SCS), gross primary
productivity (GPP), and elevation across high-latitude regions.

> **Korean summary / 한글 설명:** see [`README_ko.md`](README_ko.md).
> Original source-file provenance for every script is documented there.

---

## Repository structure

```
code/
├── 00_ml_pipeline/       ML pipeline (Jupyter notebooks) — the core SAT product
│   ├── 01_station_data_prep.ipynb    Extract MODIS + station training data
│   ├── 02_station_map.ipynb          Study-region / station map
│   ├── 03_ML_training.ipynb          Train RF / LightGBM / XGBoost (12 monthly models)
│   ├── 04_CRU_reference.ipynb        Process CRU reference climatology
│   ├── 06_ML_apply_to_MODIS.ipynb    Apply models to full MODIS grid → SAT product
│   └── 09_constraint_validation.ipynb  Validate against CRU gridded data
│
├── 01_data_pipeline/     Raw satellite / reanalysis → gridded arrays
│   ├── modis_1km_monthly.py          MODIS HDF (SC/GPP/FPAR) → monthly mean/trend arrays
│   ├── modis_water_reservoir.py      Lake/water-reservoir (MOD28C3) parser
│   ├── reanalysis_mean_trend.py      ERA5 / MERRA-2 / CPC / GHCN-CAMS mean & trend
│   ├── snowcover_sce_scs_calc.py     MODIS snow → SCE/SCS (snow end/start day) per year
│   ├── snowcover_trend_calc.py       SCE/SCS mean & trend
│   ├── start_of_season_calc.py       Start-of-season (SOS) from 8-day GPP
│   ├── gpp_max_{1,2,3}_*.py          GPP-max pipeline (align → max/trend → maps)
│   ├── sat_landmask.py               Apply land mask to SAT tiles
│   ├── regrid_to_{0.05,0.25,0.5}deg.py   Regridding utilities
│   └── run_GPP_max.sh, run_SCE.sh    Pipeline runners
│
├── 02_figure_prep/       Intermediate dataset builder for Figure 3
│
├── 10_main_figures/      Main paper figures
│   ├── fig1_polar_ML_CRU_mean_trend.py
│   ├── fig2_summer_ML_CRU_diff.py
│   ├── fig3_pattern_correlation.py   (+ fig3_meanfield_maps.py)
│   └── fig4_violin_by_altitude.py
│
├── 11_supp_figures/      Supplementary figures S3–S9
│
├── 20_temperature/       Additional temperature analyses
├── 21_lst/               Day/night land-surface-temperature analyses
├── 22_snow_interaction/  Snow ↔ temperature / GPP interaction; SCE/SCS maps
├── 23_gpp_vegetation/    GPP / FPAR analyses
└── 24_station_validation/  Station extraction & validation
```

## Requirements

Python 3.9+. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`,
`matplotlib`, `seaborn`, `cartopy`, `netCDF4`, `xarray`, `pyhdf`.

## How to run

The scripts expect the input datasets under a `data/` (and `RAW/`) directory that
is a **sibling of this `code/` directory** (i.e. `../data`, `../RAW` relative to the
repository root). Each script automatically sets its working directory to the
repository root on start, so they can be launched from anywhere, e.g.:

```bash
python 10_main_figures/fig4_violin_by_altitude.py
bash   01_data_pipeline/run_SCE.sh
```

Recommended order: `00_ml_pipeline` (produce the SAT product) →
`01_data_pipeline` → `02_figure_prep/build_figure3_datasets.py` →
`10_main_figures` / `11_supp_figures`.

External archive paths (ETOPO 2022 topography, the raw MODIS archive) are set at
the top of the relevant scripts — edit them to point to your local copies.

## Data availability

All primary datasets are publicly available from their original providers and are
**not redistributed here**:

| Dataset | Source |
|---|---|
| MODIS LST (MOD11A2 / MYD11A2), NDVI (MOD13A3 / MYD13A3), land cover (MCD12Q1) | NASA Earthdata / LP DAAC |
| MODIS GPP (MOD17A2HGF, C6.1) | NASA Earthdata / LP DAAC |
| MODIS snow cover (MOD10A2, C6.1) → SCE | NSIDC DAAC |
| CRU TS v4.06 station & gridded temperature | Climatic Research Unit, University of East Anglia |
| ERA5 reanalysis | Copernicus Climate Change Service (C3S) / ECMWF |
| MERRA-2 | NASA GMAO |
| CPC & GHCN-CAMS land-surface air temperature | NOAA Physical Sciences Laboratory |
| ETOPO 2022 global relief (topography) | NOAA NCEI |

The in-situ station air-temperature observations used to train and validate the
model are from the CRU station archive (Climatic Research Unit, University of East
Anglia). The ML-reconstructed 1-km SAT product generated in this study is archived
at *[Zenodo/figshare DOI — fill in]*.

## Code availability & citation

This code is archived at Zenodo: **https://doi.org/10.5281/zenodo.21120015**
and developed at https://github.com/SeoyeonLee-rosee/arctic-1km-sat. Please cite the
associated manuscript and the archived code release.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
