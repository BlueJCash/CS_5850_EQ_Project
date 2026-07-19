"""Figure builders for the seismic analysis project.

Every function here *returns* a figure -- a Plotly ``Figure``, a Matplotlib
``Figure`` or a Folium ``Map``.  None of them call ``plt.show()`` or
``display()``, which is what lets the same code serve the Streamlit app and the
notebook.

Matplotlib figures are built through ``Figure()`` rather than ``plt.subplots()``
on purpose.  Pyplot keeps every figure it creates in a global registry until
``plt.close()`` is called, so under Streamlit -- which re-runs this module's
functions on each interaction -- pyplot figures accumulate and leak.  Figures
constructed directly are unregistered and freed as soon as the caller drops
them.
"""

from functools import lru_cache
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram

matplotlib.use("Agg")  # headless: we only ever return figures, never show them

DATA_DIR = Path(__file__).parent
COUNTRIES_SHP = DATA_DIR / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
STATES_SHP = DATA_DIR / "ne_110m_admin_1_states_provinces" / "ne_110m_admin_1_states_provinces.shp"

#: Ceiling on points sent to the browser in one figure.  Plotly holds every
#: point as JSON in the page, so an unbounded scatter is a browser-side memory
#: problem even when the server copes fine.
MAX_PLOT_POINTS = 20_000

#: Depth is stored positive-downwards; plots negate it so the z axis reads
#: naturally with the surface at zero.
_DEPTH_LABEL = "Depth (km, below surface)"


def _scene(city):
    return dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title=_DEPTH_LABEL,
        aspectmode="cube",
    )


def _city_marker(fig, city, lat, lon):
    fig.add_trace(go.Scatter3d(
        x=[lon], y=[lat], z=[0],
        mode="markers", name=city,
        marker=dict(size=8, color="black", symbol="diamond"),
    ))
    return fig


def _layout(fig, title, city):
    fig.update_layout(
        title=title,
        scene=_scene(city),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )
    return fig


# --------------------------------------------------------------------------
# 3D scatter plots
# --------------------------------------------------------------------------

def plot_3d_clusters(events, labels, city, lat, lon, title=None):
    """Interactive 3D scatter of events coloured by cluster label."""
    fig = px.scatter_3d(
        events.assign(cluster=[str(int(v)) for v in labels]),
        x="lon", y="lat", z=-events["depth_km"],
        color="cluster", hover_data=["mag", "time", "depth_km"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=4))
    _city_marker(fig, city, lat, lon)
    return _layout(fig, title or f"Hierarchical clusters near {city}", city)


def plot_kmeans(events, labels, centroids, city, lat, lon):
    """3D scatter of k-means clusters, with centroids marked."""
    fig = plot_3d_clusters(events, labels, city, lat, lon,
                           title=f"K-means clusters near {city}")
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=-centroids[:, 2],
        mode="markers", name="Centroids",
        marker=dict(size=7, color="green", symbol="x"),
    ))
    return fig


def thin(frame, limit=MAX_PLOT_POINTS):
    """Evenly sample ``frame`` down to ``limit`` rows, preserving order.

    Plotly serialises every point into the page, so very large catalogs are a
    browser memory problem regardless of how well the server handles them.
    """
    if len(frame) <= limit:
        return frame
    step = int(np.ceil(len(frame) / limit))
    return frame.iloc[::step]


def plot_progression(splits, city, lat, lon):
    """One animated 3D figure stepping through successive time windows.

    Replaces the six separate static plots the notebook used to emit.
    """
    # Every frame lives in the figure at once, so the budget is shared out
    # across the windows rather than applied to each one.
    per_window = max(MAX_PLOT_POINTS // max(len(splits), 1), 500)
    splits = [thin(s, per_window) for s in splits]

    depths = np.concatenate([-s["depth_km"].to_numpy() for s in splits])
    lons = np.concatenate([s["lon"].to_numpy() for s in splits])
    lats = np.concatenate([s["lat"].to_numpy() for s in splits])

    def _label(split):
        return (f"{split['time'].iloc[0]:%Y-%m-%d} to "
                f"{split['time'].iloc[-1]:%Y-%m-%d}")

    def _trace(split):
        return go.Scatter3d(
            x=split["lon"], y=split["lat"], z=-split["depth_km"],
            mode="markers", name="Events",
            marker=dict(size=4, color=split["mag"], colorscale="Turbo",
                        cmin=float(min(s["mag"].min() for s in splits)),
                        cmax=float(max(s["mag"].max() for s in splits)),
                        colorbar=dict(title="Magnitude")),
            text=[f"M{m:.1f}" for m in split["mag"]],
        )

    frames = [go.Frame(data=[_trace(s)], name=_label(s)) for s in splits]
    fig = go.Figure(data=[_trace(splits[0])], frames=frames)
    _city_marker(fig, city, lat, lon)

    # Hold the axes fixed across frames so movement reads as real change.
    pad = 0.05
    fig.update_layout(
        scene=dict(
            **_scene(city),
            xaxis=dict(range=[lons.min() - pad, lons.max() + pad]),
            yaxis=dict(range=[lats.min() - pad, lats.max() + pad]),
            zaxis=dict(range=[depths.min(), max(depths.max(), 0)]),
        ),
        title=f"Seismic progression near {city}",
        height=680,
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.05, xanchor="left",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Window: "),
            steps=[dict(label=f.name, method="animate",
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))])
                   for f in frames],
        )],
    )
    return fig


# --------------------------------------------------------------------------
# Dendrogram and forecasts
# --------------------------------------------------------------------------

def plot_dendrogram(linked):
    """Hierarchical clustering dendrogram as a Matplotlib figure."""
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    # truncate_mode keeps the drawing bounded: a full tree over tens of
    # thousands of leaves produces an unreadable figure and a large one.
    dendrogram(linked, ax=ax, no_labels=True,
               truncate_mode="lastp" if len(linked) > 500 else None, p=100)
    ax.set_title("Hierarchical clustering dendrogram (Ward's method)")
    ax.set_xlabel("Event")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


def plot_forecast(history, forecast, backcast, title, period_label="period"):
    """Historical counts with forecast and backcast extensions."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Historical",
                             mode="lines+markers", line=dict(color="#3366cc")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                             mode="lines+markers", line=dict(color="#dc3912", dash="dash")))
    fig.add_trace(go.Scatter(x=backcast.index, y=backcast.values, name="Backcast",
                             mode="lines+markers", line=dict(color="#109618", dash="dot")))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Events per {period_label}",
        height=470,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_importances(importances):
    """Decision tree feature importances as a horizontal bar chart."""
    ordered = importances.sort_values()
    fig = go.Figure(go.Bar(x=ordered.values, y=ordered.index, orientation="h",
                           marker_color="#3366cc"))
    fig.update_layout(title="Feature importance", xaxis_title="Importance",
                      height=320, margin=dict(l=0, r=0, t=50, b=0))
    return fig


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------

def heatmap(events, city, lat, lon, maxradius=1.0, topo=False):
    """Folium heat map of event density, optionally on topographic tiles."""
    m = folium.Map(location=[lat, lon], zoom_start=8)
    if topo:
        folium.TileLayer("OpenTopoMap", attr="Map tiles by OpenTopoMap").add_to(m)

    HeatMap(events[["lat", "lon"]].values.tolist(), radius=10).add_to(m)
    folium.Circle(
        location=(lat, lon),
        radius=maxradius * 111_000,  # one degree of latitude is ~111 km
        color="red", fill=False,
        popup=f"Search area: {maxradius}deg (~{maxradius * 111:.0f} km)",
    ).add_to(m)
    folium.Marker([lat, lon], tooltip=city,
                  icon=folium.Icon(color="black", icon="star")).add_to(m)
    return m


@lru_cache(maxsize=2)
def _boundaries(level):
    """Natural Earth boundaries, read once and reused.

    Only two small (~200 kB) frames can ever be held, so the cache is bounded.
    """
    if level == "state":
        return gpd.read_file(STATES_SHP), "name"
    return gpd.read_file(COUNTRIES_SHP), "NAME"


def region_names(level="country"):
    """Names available in the Natural Earth shapefiles, for the region picker."""
    world, column = _boundaries(level)
    return sorted(world[column].dropna().unique())


def choropleth(events, region_name, level="country"):
    """Scatter of events over the world map, zoomed to the chosen region."""
    world, column = _boundaries(level)
    region = world[world[column] == region_name]

    if region.empty:
        raise ValueError(f"{region_name!r} is not present in the {level} shapefile.")

    # points_from_xy is vectorised; building a Python list of Point objects
    # allocates one object per event and is needlessly heavy on large catalogs.
    # Only the two coordinate columns are carried over.
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(events["lon"], events["lat"]),
        crs=world.crs,
    )

    fig = Figure(figsize=(11, 7))
    ax = fig.subplots()
    world.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.4)
    region.plot(ax=ax, color="lightblue", edgecolor="steelblue", linewidth=0.6)
    points.plot(ax=ax, color="red", markersize=12, alpha=0.7)

    minx, miny, maxx, maxy = region.total_bounds
    ax.set_xlim(minx - 2, maxx + 2)
    ax.set_ylim(miny - 2, maxy + 2)
    ax.set_title(f"Seismic events across {region_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    return fig
