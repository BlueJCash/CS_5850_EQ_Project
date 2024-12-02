from obspy.clients.fdsn import Client
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings


def get_geolocation(city="Challis, ID"):
    """Fetch the latitude and longitude of a city."""
    geolocator = Nominatim(user_agent="TomPEQProject")
    try:
        loc = geolocator.geocode(city)
        if loc:
            return loc.latitude, loc.longitude
        else:
            print(f"City {city} not found.")
            return None
    except Exception as e:
        print(f"Error fetching geolocation for {city}: {e}")
        return None


def get_earthquake_data(client="USGS", city="Challis, ID", maxradius=1, starttime="1900-10-01", endtime="2024-10-11", minmagnitude=2.5):
    """Fetch earthquake events around a city."""
    lat, lon = get_geolocation(city)
    if lat is None or lon is None:
        return None

    try:
        catalog = client.get_events(starttime=starttime, 
                                endtime=endtime, 
                                minmagnitude=minmagnitude,
                                eventtype="earthquake",
                                latitude=lat,
                                longitude=lon,
                                maxradius=maxradius
                                )
        return catalog
    except Exception as e:
        print(f"Error fetching events: {e}")
        return None


def process_earthquake_data(catalog):
    """Process the fetched earthquake events and extract location information."""
    locs = []
    if catalog:
        for event in catalog:
            temp = event.origins[0]
            if temp.longitude is not None and temp.latitude is not None and temp.depth is not None:
                location = [temp.longitude, temp.latitude, temp.depth, temp.time, event.magnitudes[0].mag]  
                locs.append(location)
        max_depth = max(locs, key=lambda x: x[2])[2]

        locs = np.array(locs)
        locs = locs[locs[:, 3].argsort()]
        
        return locs, max_depth
    else:
        return None


def normalize_depth(locs):
    """Normalize the depth values."""
    scaler = MinMaxScaler()
    locs[:, 2] = -scaler.fit_transform(locs[:, 2].reshape(-1, 1)).flatten()
    return locs


def perform_clustering(locs):
    """Perform hierarchical clustering on earthquake locations."""
    linked = linkage(locs, method='ward')
    return linked


def perform_labeling(linked, k):
    """Assign labels to clusters."""
    labels = fcluster(linked, k, criterion='maxclust')
    return labels


def plot_dendrogram(linked):
    """Plot the hierarchical clustering dendrogram."""
    plt.figure(figsize=(10, 7))
    dendro = dendrogram(linked)
    color_dict = {color for color in dendro['color_list']}
    num_colors = len(color_dict)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Event index')
    plt.ylabel('Distance')
    plt.show()
    
    return num_colors - 1


def plot_3d_clusters(locs, labels, city, max_depth, elev, azim):
    """Plot the 3D scatter plot of earthquake locations and clusters."""
    fig = plt.figure(figsize=(10,8))
    lat, lon = get_geolocation(city=city)

    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2]*max_depth, c=labels, cmap='jet', marker='o', label='Events')
    ax.scatter(lon, lat, 0, c='black', marker='*', s=200, label='City')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (km)')

    ax.set_title(f'Hierarchical Clustering of Earthquake Locations for {city} (Latitude, Longitude, Depth) ')
    ax.legend()

    ax.view_init(elev=elev, azim=azim)

    plt.show()


def predict_Seismic_RF(city, locs, duration=30, freq='YE'):
    """Predict the number of earthquake events in a future/Past time span using Random Forest."""
    
    event_times = [time.datetime for time in locs[:, 3]] # Convert to datetime
    
    event_counts = pd.Series(1, index=pd.to_datetime(event_times)).resample(freq).count() # Count number of events per freq
    
    event_counts_df = pd.DataFrame(event_counts)
    event_counts_df['year'] = event_counts_df.index.year
    event_counts_df['month'] = event_counts_df.index.month
    event_counts_df['day'] = event_counts_df.index.day
    
    X = event_counts_df[['year', 'month', 'day']]
    y = event_counts_df[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    future_dates = pd.date_range(event_counts.index[-1], periods=duration+1, freq=freq)[1:]
    future_features = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day
    })
    
    forecast = rf_model.predict(future_features)
    
    past_dates = pd.date_range(event_counts.index[0] - pd.DateOffset(years=duration), periods=duration+1, freq=freq)[1:]
    backcast_features = pd.DataFrame({
        'year': past_dates.year,
        'month': past_dates.month,
        'day': past_dates.day
    })
    backcast = rf_model.predict(backcast_features)
    
    plt.figure(figsize=(10, 8))
    plt.plot(event_counts.index, event_counts, label='Historical Data', color='blue')  # Historical data plot
    plt.plot(future_dates, forecast, label='Forecasted Data', color='red')  # Forecast data plot
    plt.plot(past_dates, backcast, label='Backcasted Data', color='green')
    plt.title(f'Event Count Forecast for {city} using RF')
    plt.xlabel('Date')
    plt.ylabel('Event Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # For Readability
    print(f"\nPredicted Number of Events for {city} in the prior {duration} {freq}s:")
    for date, count in zip(past_dates, backcast):
        print(f"{date.strftime('%Y-%m-%d')}    {count:.2f}")
    print(f"Predicted Number of Events for {city} in the next {duration} {freq}s:")
    for date, count in zip(future_dates, forecast):
        print(f"{date.strftime('%Y-%m-%d')}    {count:.2f}")


def predict_Seismic_ARIMA(city, locs, duration=30, freq='YE'):
    """Predict the number of earthquake events in a future time span using ARIMA."""
    
    event_times = [time.datetime for time in locs[:, 3]] # Convert to datetime
    
    event_counts = pd.Series(1, index=pd.to_datetime(event_times)).resample(freq).count() # Count number of events per freq
    
    event_counts_reversed = event_counts[::-1] # For Backcasting
    
    warnings.filterwarnings("ignore") # Warning caused by performing back prediction and non-stationary
    model = ARIMA(event_counts, order=(5, 1, 0)) # Create ARIMA model
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=duration) # Create forecast
    future_dates = pd.date_range(event_counts.index[-1], periods=duration+1, freq=freq)[1:] # Starting at end of historical data
    
    model = ARIMA(event_counts_reversed, order=(5, 1, 0)) # Create ARIMA model
    model_fit = model.fit()
    backcast = model_fit.forecast(steps=duration) # Create backcast 
    past_dates = pd.date_range(event_counts.index[0] - pd.DateOffset(years=duration), periods=duration+1, freq=freq)[1:]
    warnings.filterwarnings("default")
    
    forecast[forecast < 0] = 0
    backcast[backcast < 0] = 0
    
    plt.figure(figsize=(10, 8))
    plt.plot(event_counts.index, event_counts, label='Historical Data', color='blue')  # Historical data plot
    plt.plot(future_dates, forecast, label='Forecasted Data', color='red')  # Forecast data plot
    plt.plot(past_dates, backcast, label='Backcasted Data', color='green')
    plt.title(f'Event Count Forecast for {city} using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Event Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # For Readability
    print(f"\nPredicted Number of Events for {city}")
