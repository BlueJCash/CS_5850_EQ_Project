# Small Seismic Event Analysis Project

By Thomas Prouty

## Overview

The goal of the project was to investigate the effectiveness of different visualization and prediction methods for displaying and understanding historical seismic data. This project provides tools for performing seismic analysis, visualization, clustering, and predictions for earthquake data. Specifically, the code supports hierarchical clustering, heatmap visualization, k-means clustering, and forecasting/backcasting using Random Forests and ARIMA models.

Everything is driven from a web app — pick a city and a set of parameters in the sidebar, press **Run analysis**, and the results appear across six tabs.

## Table of Contents

* Quick Start
* Features
* The App
* Project Layout
* Using the Modules Directly
* Data Source
* Contact Info

## Quick Start

```powershell
# 1. Create the environment (Python 3.14; every dependency ships as a wheel)
uv venv --python 3.14 .venv
uv pip install -r requirements.txt

# 2. Launch the app
.\run_app.ps1
```

The app opens at <http://localhost:8501>. Without `uv`, any Python 3.11+ works:
`python -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`
and `.venv\Scripts\streamlit run app.py`.

An internet connection is required — event data is fetched live from the USGS
catalog and place names are resolved through OpenStreetMap's Nominatim service.

## Features

1. Modularity in location calling
2. Hierarchical Clustering using Ward's Method
3. K-Means Clustering
4. Choropleth Scatterplot
5. Heat Map, with or without topography
6. Animated progression of activity through time
7. Forecasting and Backcasting via Random Forest and ARIMA
8. Trait or Important Feature Determination via Decision Tree

## The App

All parameters live in the sidebar; nothing recomputes until **Run analysis** is
pressed.

| Parameter | Meaning |
| --- | --- |
| City | Focal point of the analysis. Add a state or country to disambiguate. |
| Search radius | How far around the focal point to gather data, in degrees (1° ≈ 111 km). |
| Date range | Period to request data for. |
| Magnitude range | Minimum and maximum magnitude to include. |
| Aggregation period | Year, quarter or month — the bucket size for forecasting. |
| Forecast length | How many periods to project beyond each end of the record. |
| Clusters | Number of clusters; `0` reads the count off the dendrogram automatically. |
| Decision tree depth / min samples per leaf | Shape of the regression tree. |
| Choropleth region | Boundary to zoom the regional map to; blank detects it from the city. |

Results are cached on the query parameters, so adjusting a cluster count or tree
depth re-renders immediately instead of re-downloading from USGS.

**Tabs:** Overview (metrics, catalog table, CSV download) · Clustering
(dendrogram, hierarchical and k-means 3D plots) · Maps (heat map, topographic
heat map, choropleth) · Progression (animated 3D) · Forecasting (Random Forest
and ARIMA) · Decision tree.

The 3D plots are interactive — drag to rotate, scroll to zoom.

## Project Layout

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit UI. The only file that renders anything. |
| `eq_analysis.py` | Data retrieval, clustering, forecasting, decision tree. Returns values only. |
| `eq_plots.py` | Figure builders. Returns Plotly / Matplotlib / Folium objects. |
| `EQ_Analysis_Run.ipynb` | Notebook walkthrough of the same analysis, step by step. |
| `requirements.txt` | Pinned dependencies, verified on Python 3.14. |
| `run_app.ps1` | Launcher for the app. |

The Natural Earth shapefiles used by the choropleth are included in the repo:

```text
ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp
ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp
```

## Using the Modules Directly

Both modules are importable, so the analysis can be scripted without the UI:

```python
import eq_analysis as eq
import eq_plots as plots

events = eq.fetch_events("Santorini, Greece", maxradius=1.0,
                         starttime="2000-10-01", endtime="2024-10-11")
lat, lon, country = eq.get_geolocation("Santorini, Greece")

linked = eq.perform_clustering(events)
k = eq.suggest_k(linked)
labels = eq.perform_labeling(linked, k)

plots.plot_3d_clusters(events, labels, "Santorini, Greece", lat, lon).show()
```

`fetch_events` returns a DataFrame with columns `lon`, `lat`, `depth_km`, `time`
and `mag`. Anything that cannot be satisfied — an unknown city, a query matching
no events, a catalog too large to cluster — raises `eq.AnalysisError` carrying a
message suitable for showing to a user.

### Notes on limits

Hierarchical clustering builds a pairwise distance matrix that grows with the
square of the event count, so `perform_clustering` refuses catalogs above
`eq.MAX_CLUSTER_EVENTS` (10,000) rather than exhausting memory. Narrow the
radius, shorten the date range, or raise the minimum magnitude. Plots are
likewise thinned to `plots.MAX_PLOT_POINTS` before being sent to the browser.

The decision tree is rendered from its DOT source directly in the browser, so
the app does **not** require the Graphviz binary. The notebook's final cell does,
and will tell you so if it is missing.

## Data Source

USGS Earthquake Catalog, via the FDSN web service. Geocoding by OpenStreetMap Nominatim.

## Contact Info

If there are any questions please message Tom Prouty via Canvas and I will work to answer them as soon as possible.
