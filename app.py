"""Streamlit front end for the small seismic event analysis project.

Run it with::

    streamlit run app.py

Every parameter that used to be a hardcoded variable inside ``main()`` is a
control in the sidebar.  Results are cached on the query parameters, so
adjusting a cluster count or tree depth re-renders without re-downloading the
catalog from USGS.
"""

import datetime as dt

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import eq_analysis as eq
import eq_plots as plots

st.set_page_config(page_title="Seismic Event Analysis", page_icon="~", layout="wide")

# Natural Earth spells a few countries differently from Nominatim.
COUNTRY_ALIASES = {
    "United States": "United States of America",
    "Russian Federation": "Russia",
    "Czechia": "Czech Republic",
}


# --------------------------------------------------------------------------
# Cached wrappers
#
# Each wrapper is keyed on its arguments, so only the stages whose inputs
# actually changed are recomputed on a rerun.
#
# Every cache is given an explicit `max_entries`.  Streamlit's default is
# unbounded, which would mean that idly exploring cities grows the process
# without limit -- each entry holds a full event catalog or a fitted model.
# The download caches also carry a TTL so a long-running session eventually
# picks up new events from USGS.
# --------------------------------------------------------------------------

_DAY = 60 * 60 * 24

@st.cache_data(show_spinner="Locating city...", max_entries=64, ttl=_DAY)
def geocode(city):
    return eq.get_geolocation(city)


@st.cache_data(show_spinner="Downloading events from USGS...", max_entries=8, ttl=_DAY)
def load_events(city, maxradius, start, end, minmag, maxmag):
    return eq.fetch_events(city, maxradius, start, end, minmag, maxmag)


@st.cache_data(show_spinner="Clustering...", max_entries=4)
def cluster(events, k_requested):
    linked = eq.perform_clustering(events)
    k = k_requested or eq.suggest_k(linked)
    return linked, k, eq.perform_labeling(linked, k)


@st.cache_data(show_spinner="Running k-means...", max_entries=4)
def kmeans(events, k):
    return eq.kmeans_clusters(events, k)


@st.cache_data(show_spinner="Fitting random forest...", max_entries=4)
def rf(events, duration, freq):
    return eq.forecast_rf(events, duration, freq)


@st.cache_data(show_spinner="Fitting ARIMA...", max_entries=4)
def arima(events, duration, freq):
    return eq.forecast_arima(events, duration, freq)


@st.cache_data(show_spinner="Growing decision tree...", max_entries=4)
def tree(events, max_depth, min_samples_leaf):
    return eq.decision_tree(events, max_depth, min_samples_leaf)


@st.cache_data(max_entries=2)
def regions(level):
    return plots.region_names(level)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Analysis settings")

with st.sidebar.form("settings"):
    st.subheader("Where and when")
    city = st.text_input("City", value="Santorini, Greece",
                         help="Any place name. Add a state or country to disambiguate.")
    maxradius = st.slider("Search radius (degrees)", 0.1, 10.0, 1.0, 0.1,
                          help="1 degree is roughly 111 km.")
    date_range = st.date_input(
        "Date range",
        value=(dt.date(2000, 10, 1), dt.date(2024, 10, 11)),
        min_value=dt.date(1900, 1, 1), max_value=dt.date.today(),
    )
    minmag, maxmag = st.slider("Magnitude range", 0.0, 10.0, (2.5, 6.0), 0.1)

    st.subheader("Models")
    freq_label = st.selectbox("Aggregation period", list(eq.FREQUENCIES), index=0)
    duration = st.slider("Forecast / backcast length (periods)", 1, 50, 20)
    clusters = st.number_input("Clusters (0 = choose automatically)", 0, 20, 0)
    max_tree_depth = st.slider("Decision tree max depth", 1, 15, 6)
    min_samples_leaf = st.slider("Decision tree min samples per leaf", 1, 100, 10)

    st.subheader("Choropleth region")
    region_level = st.radio("Boundary level", ["country", "state"], horizontal=True)
    region_override = st.text_input(
        "Region name (blank = detect from city)", value="",
        help="Must match a Natural Earth name, e.g. 'Greece' or 'Idaho'.",
    )

    submitted = st.form_submit_button("Run analysis", type="primary",
                                      width="stretch")

if submitted:
    st.session_state["ran"] = True

if not st.session_state.get("ran"):
    st.title("Seismic Event Analysis")
    st.markdown(
        "Explore historical earthquake activity around any city using the USGS "
        "catalog: clustering, heat maps, forecasting and feature analysis.\n\n"
        "Set the parameters in the sidebar, then press **Run analysis**."
    )
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    st.error("Please pick both a start and an end date.")
    st.stop()

freq = eq.FREQUENCIES[freq_label]
period_label = freq_label.lower()

# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

try:
    lat, lon, country = geocode(city)
    events = load_events(city, maxradius, str(start), str(end), minmag, maxmag)
except eq.AnalysisError as exc:
    st.error(str(exc))
    st.stop()

region_name = region_override.strip() or COUNTRY_ALIASES.get(country, country)

st.title(f"Seismic activity near {city}")
top = st.columns(4)
top[0].metric("Events", f"{len(events):,}")
top[1].metric("Magnitude range", f"{events['mag'].min():.1f} - {events['mag'].max():.1f}")
top[2].metric("Median depth", f"{events['depth_km'].median():.1f} km")
top[3].metric("Span", f"{events['time'].min():%Y} - {events['time'].max():%Y}")

tabs = st.tabs(["Overview", "Clustering", "Maps", "Progression",
                "Forecasting", "Decision tree"])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with tabs[0]:
    st.caption(f"Centred on {lat:.4f}, {lon:.4f}"
               + (f" ({country})" if country else "")
               + f" - radius {maxradius}deg (~{maxradius * 111:.0f} km).")

    left, right = st.columns(2)
    with left:
        st.subheader("Events over time")
        st.bar_chart(events.set_index("time").resample(freq).size(),
                     y_label=f"Events per {period_label}")
    with right:
        st.subheader("Magnitude distribution")
        st.bar_chart(events["mag"].round(1).value_counts().sort_index(),
                     y_label="Events")

    st.subheader("Event catalog")
    st.dataframe(events, width="stretch", hide_index=True)
    st.download_button("Download as CSV", events.to_csv(index=False),
                       file_name=f"events_{city.replace(', ', '_')}.csv",
                       mime="text/csv")

# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

with tabs[1]:
    try:
        linked, k, labels = cluster(events, int(clusters))
    except eq.AnalysisError as exc:
        st.warning(str(exc))
    else:
        if not clusters:
            st.info(f"Cluster count chosen automatically from the dendrogram: **{k}**.")

        st.subheader("Hierarchical clustering (Ward's method)")
        st.plotly_chart(plots.plot_3d_clusters(events, labels, city, lat, lon),
                        width="stretch")
        st.caption("Drag to rotate, scroll to zoom.")

        with st.expander("Dendrogram"):
            st.pyplot(plots.plot_dendrogram(linked))

        st.subheader("K-means clustering")
        try:
            km_labels, centroids = kmeans(events, k)
        except eq.AnalysisError as exc:
            st.warning(str(exc))
        else:
            st.plotly_chart(plots.plot_kmeans(events, km_labels, centroids, city, lat, lon),
                            width="stretch")

# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Event density")
    topo = st.toggle("Topographic tiles", value=False)
    st_folium(plots.heatmap(events, city, lat, lon, maxradius, topo=topo),
              height=520, use_container_width=True, returned_objects=[])

    st.subheader("Regional context")
    try:
        st.pyplot(plots.choropleth(events, region_name, region_level))
    except ValueError:
        st.warning(
            f"{region_name!r} was not found in the {region_level} boundaries. "
            "Enter a name from the list below in the sidebar."
        )
        st.write(regions(region_level))

# --------------------------------------------------------------------------
# Progression
# --------------------------------------------------------------------------

with tabs[3]:
    st.subheader("How activity moved over time")
    n_splits = st.slider("Number of time windows", 2, 12, 6)
    if len(events) < n_splits:
        st.warning("Not enough events to split into that many windows.")
    else:
        splits = eq.split_by_date(events, n_splits)
        st.plotly_chart(plots.plot_progression(splits, city, lat, lon),
                        width="stretch")
        st.caption("Press Play, or drag the slider to step through the windows. "
                   "Points are coloured by magnitude.")

# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

with tabs[4]:
    st.caption(f"Predicting the number of events per {period_label}, "
               f"{duration} {period_label}s beyond each end of the record.")

    for name, fn in (("Random forest", rf), ("ARIMA", arima)):
        st.subheader(name)
        try:
            history, forecast, backcast = fn(events, duration, freq)
        except eq.AnalysisError as exc:
            st.warning(str(exc))
            continue
        except Exception as exc:  # ARIMA can fail to converge on sparse series
            st.warning(f"{name} could not fit this series: {exc}")
            continue

        st.plotly_chart(
            plots.plot_forecast(history, forecast, backcast,
                                f"Event count {name.lower()} for {city}",
                                period_label),
            width="stretch",
        )
        with st.expander(f"{name} predicted values"):
            st.dataframe(
                pd.DataFrame({
                    "Backcast": backcast.round(2),
                    "Forecast": forecast.round(2),
                }).rename_axis("Period"),
                width="stretch",
            )

# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------

with tabs[5]:
    st.subheader("What predicts magnitude?")
    try:
        _model, dot_source, importances = tree(events, max_tree_depth, min_samples_leaf)
    except eq.AnalysisError as exc:
        st.warning(str(exc))
    else:
        st.plotly_chart(plots.plot_importances(importances), width="stretch")
        st.caption("Regression tree over event time and location, predicting magnitude.")
        st.graphviz_chart(dot_source, width="stretch")
