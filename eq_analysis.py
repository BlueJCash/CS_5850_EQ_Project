"""Data retrieval, preparation and modelling for the seismic analysis project.

Nothing in this module draws or displays anything -- every function returns
plain values (DataFrames, arrays, fitted models).  Figure building lives in
``eq_plots`` and rendering lives in ``app.py``.

The canonical representation of a dataset is an ``events`` DataFrame:

    lon       float64          degrees east
    lat       float64          degrees north
    depth_km  float64          kilometres below the surface (positive)
    time      datetime64[ns]   event origin time (UTC)
    mag       float64          reported magnitude

Column names -- rather than positional indices into a numpy array -- are what
keep latitude and longitude from being transposed downstream.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from obspy.clients.fdsn import Client
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from statsmodels.tsa.arima.model import ARIMA

FEATURES = ["year", "month", "day", "lat", "lon", "depth_km"]

#: Resampling frequencies offered in the UI, mapped to pandas offset aliases.
FREQUENCIES = {"Year": "YE", "Quarter": "QE", "Month": "ME"}

#: Hierarchical clustering needs a pairwise distance matrix, whose size grows
#: with the square of the event count.  10,000 events costs roughly 400 MB;
#: beyond this ceiling ``perform_clustering`` refuses rather than thrashing.
MAX_CLUSTER_EVENTS = 10_000


class AnalysisError(Exception):
    """Raised when a request cannot be satisfied (bad city, no events, ...).

    The UI catches this and shows the message, rather than a traceback.
    """


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

@lru_cache(maxsize=128)
def get_geolocation(city="Challis, ID"):
    """Return ``(lat, lon, country)`` for a place name.

    Cached because a single analysis asks for the same city many times over and
    Nominatim's usage policy allows only one request per second.
    """
    geolocator = Nominatim(user_agent="TomPEQProject")
    try:
        # language="en" matters: without it Nominatim answers in the local
        # language ("Ellada"), which will not match the Natural Earth names
        # used by the choropleth.
        loc = geolocator.geocode(city, addressdetails=True, language="en")
    except Exception as exc:
        raise AnalysisError(f"Could not reach the geocoding service: {exc}") from exc

    if loc is None:
        raise AnalysisError(f"City {city!r} not found. Try adding a state or country.")

    country = (loc.raw.get("address") or {}).get("country")
    return loc.latitude, loc.longitude, country


def fetch_events(city="Challis, ID", maxradius=1.0, starttime="1900-10-01",
                 endtime="2024-10-11", minmagnitude=2.5, maxmagnitude=6.0):
    """Download earthquakes around ``city`` and return an ``events`` DataFrame.

    ``maxradius`` is in degrees, as the FDSN service expects.
    """
    lat, lon, _country = get_geolocation(city)

    try:
        catalog = Client("USGS").get_events(
            starttime=str(starttime),
            endtime=str(endtime),
            minmagnitude=minmagnitude,
            maxmagnitude=maxmagnitude,
            eventtype="earthquake",
            latitude=lat,
            longitude=lon,
            maxradius=maxradius,
        )
    except Exception as exc:
        # The FDSN client raises a bare exception for "no data matched" too.
        raise AnalysisError(
            f"No earthquakes found for {city} with these settings, or the USGS "
            f"service could not be reached ({exc})."
        ) from exc

    rows = []
    for event in catalog:
        origin = event.origins[0]
        if origin.longitude is None or origin.latitude is None or origin.depth is None:
            continue  # drop events with incomplete location data
        magnitude = event.magnitudes[0].mag if event.magnitudes else np.nan
        rows.append((origin.longitude, origin.latitude,
                     origin.depth / 1000.0,  # obspy reports depth in metres
                     origin.time.datetime, magnitude))

    if not rows:
        raise AnalysisError(f"No usable events returned for {city}.")

    events = pd.DataFrame(rows, columns=["lon", "lat", "depth_km", "time", "mag"])
    events["time"] = pd.to_datetime(events["time"])
    return events.sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

def cluster_features(events):
    """Return the (lon, lat, normalized depth) matrix used for clustering.

    Depth is scaled to [0, 1] so that it carries comparable weight to the two
    angular coordinates.  The raw ``depth_km`` column is left untouched.
    """
    scaled = MinMaxScaler().fit_transform(events[["depth_km"]])
    return np.column_stack([events["lon"], events["lat"], scaled.flatten()])


def split_by_date(events, num_splits=6):
    """Split a time-ordered ``events`` frame into ``num_splits`` equal chunks.

    Returns a list of DataFrames.  (``np.array_split`` would strip the column
    names and hand back bare object arrays.)
    """
    ordered = events.sort_values("time").reset_index(drop=True)
    bounds = np.array_split(np.arange(len(ordered)), num_splits)
    return [ordered.iloc[idx] for idx in bounds if len(idx)]


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def perform_clustering(events):
    """Hierarchical clustering of event locations using Ward's method.

    ``linkage`` builds a condensed pairwise distance matrix, which costs
    ``n * (n - 1) / 2`` doubles -- quadratic in the number of events.  At 10,000
    events that is already ~400 MB and at 30,000 it is ~3.6 GB, so the request
    is refused rather than allowed to exhaust memory.  The old notebook never
    hit this because its radius was fixed at one degree; the app's radius slider
    makes far larger catalogs reachable.
    """
    if len(events) < 2:
        raise AnalysisError("At least two events are needed for clustering.")

    if len(events) > MAX_CLUSTER_EVENTS:
        required_gb = len(events) * (len(events) - 1) * 4 / 1024 ** 3
        raise AnalysisError(
            f"{len(events):,} events is too many to cluster hierarchically: the "
            f"pairwise distance matrix alone would need about {required_gb:.1f} GB. "
            f"Narrow the search radius, shorten the date range, or raise the "
            f"minimum magnitude to get under {MAX_CLUSTER_EVENTS:,} events."
        )

    return linkage(cluster_features(events), method="ward")


def suggest_k(linked):
    """Cluster count implied by the dendrogram's own colouring.

    ``plot_dendrogram`` used to return this as a side effect of drawing; here it
    is computed with ``no_plot=True`` so no figure is required.
    """
    colors = {c for c in dendrogram(linked, no_plot=True)["color_list"]}
    return max(len(colors) - 1, 2)


def perform_labeling(linked, k):
    """Cut the linkage tree into ``k`` flat clusters."""
    return fcluster(linked, k, criterion="maxclust")


def kmeans_clusters(events, k=3):
    """K-means over event locations. Returns ``(labels, centroids)``.

    Centroid depths are returned in kilometres so they can be plotted directly
    against the events.
    """
    if len(events) < k:
        raise AnalysisError(f"Need at least {k} events to form {k} clusters.")

    features = cluster_features(events)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)

    # Undo the depth scaling on the centroids so they share the events' axis.
    depth = events["depth_km"]
    lo, hi = depth.min(), depth.max()
    centroids = kmeans.cluster_centers_.copy()
    centroids[:, 2] = centroids[:, 2] * (hi - lo) + lo
    return kmeans.labels_, centroids


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

def _event_counts(events, freq):
    """Number of events per ``freq`` period, indexed by period end."""
    counts = events.set_index("time").resample(freq).size()
    if len(counts) < 3:
        raise AnalysisError(
            "Not enough history to forecast at this frequency. Widen the date "
            "range or choose a shorter period (e.g. Month instead of Year)."
        )
    return counts


def _past_dates(index, duration, freq):
    """``duration`` periods immediately preceding ``index[0]``, ascending."""
    return pd.date_range(end=index[0], periods=duration + 1, freq=freq)[:-1]


def _future_dates(index, duration, freq):
    """``duration`` periods immediately following ``index[-1]``, ascending."""
    return pd.date_range(start=index[-1], periods=duration + 1, freq=freq)[1:]


def forecast_rf(events, duration=20, freq="YE"):
    """Forecast and backcast event counts with a random forest.

    Returns ``(history, forecast, backcast)`` as three pandas Series.
    """
    frame = events.set_index("time")
    counts = _event_counts(events, freq)

    aggregated = frame[["lat", "lon", "depth_km"]].resample(freq).mean()
    design = aggregated.assign(
        event_count=counts,
        year=aggregated.index.year,
        month=aggregated.index.month,
        day=aggregated.index.day,
    ).dropna()

    if len(design) < 3:
        raise AnalysisError("Not enough populated periods to train a random forest.")

    X, y = design[FEATURES], design["event_count"]
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Location is not the quantity being predicted, so hold it at its mean and
    # let the date fields carry the signal.
    means = {c: X[c].mean() for c in ("lat", "lon", "depth_km")}

    def _predict(dates):
        design_matrix = pd.DataFrame(
            {"year": dates.year, "month": dates.month, "day": dates.day, **means},
            index=dates,
        )[FEATURES]
        return pd.Series(model.predict(design_matrix), index=dates)

    future = _future_dates(design.index, duration, freq)
    past = _past_dates(design.index, duration, freq)
    return counts, _predict(future), _predict(past)


def forecast_arima(events, duration=20, freq="YE", order=(5, 1, 0)):
    """Forecast and backcast event counts with ARIMA.

    The backcast is produced by fitting the same model to the time-reversed
    series, then flipping the result back into chronological order.
    """
    counts = _event_counts(events, freq)

    forecast = ARIMA(counts.values, order=order).fit().forecast(steps=duration)
    backcast = ARIMA(counts.values[::-1], order=order).fit().forecast(steps=duration)

    future = _future_dates(counts.index, duration, freq)
    past = _past_dates(counts.index, duration, freq)

    forecast = pd.Series(np.clip(forecast, 0, None), index=future)
    # The reversed fit predicts backwards in time; flip it to match ``past``.
    backcast = pd.Series(np.clip(backcast[::-1], 0, None), index=past)
    return counts, forecast, backcast


# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------

def decision_tree(events, max_tree_depth=6, min_samples_leaf=10):
    """Regress magnitude on time and location to surface influential features.

    Returns ``(model, dot_source, importances)``.  The DOT source is handed to
    the browser for rendering, so the Graphviz ``dot`` binary is not required.
    """
    design = events.assign(
        year=events["time"].dt.year,
        month=events["time"].dt.month,
        day=events["time"].dt.day,
    )

    X, y = design[FEATURES], design["mag"]
    if len(X) < 5:
        raise AnalysisError("Not enough events to fit a decision tree.")

    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeRegressor(
        max_depth=max_tree_depth, min_samples_leaf=min_samples_leaf, random_state=42
    ).fit(X_train, y_train)

    dot_source = export_graphviz(
        model, out_file=None, feature_names=FEATURES,
        filled=True, rounded=True, special_characters=True,
    )
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    return model, dot_source, importances.sort_values(ascending=False)
