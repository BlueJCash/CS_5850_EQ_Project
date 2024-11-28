from obspy.clients.fdsn import Client
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting


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
                location = [temp.longitude, temp.latitude, temp.depth, temp.time]  
                locs.append(location)
                #print(f"Event Time: {temp.time}, Magnitude: {event.magnitudes[0].mag}")
                #print(location)
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
    scatter = ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2]*max_depth, c=labels, cmap='viridis', marker='o', label='Events')
    ax.scatter(lon, lat, 0, c='black', marker='*', s=200, label='City')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (km)')

    ax.set_title(f'Hierarchical Clustering of Earthquake Locations for {city} (Latitude, Longitude, Depth)')
    ax.legend()

    ax.view_init(elev=elev, azim=azim)

    plt.show()

def perform_Seismitc_Analysis(city="Challis, ID", maxradius=1, elev=0, azim=90):   
    client = Client("USGS")
    catalog = get_earthquake_data(client=client, city=city, maxradius=maxradius)
    if catalog:
        locs, max_depth = process_earthquake_data(catalog)
        locs = normalize_depth(locs)
        
        linked = perform_clustering(locs)
        k = plot_dendrogram(linked)
        labels = perform_labeling(linked, k)

        plot_3d_clusters(locs, labels, city, max_depth, elev=elev, azim=azim)
    else:
        return None


def split_by_date(locs, num_splits=6):
    """Splits the data into equals sections"""
    splits = np.array_split(locs, num_splits)
    return splits

def show_progression(city="Challis, ID", maxradius=1, elev=0, azim=90):
    client = Client("USGS")
    catalog = get_earthquake_data(client=client, city=city, maxradius=maxradius)
    if catalog:
        locs, max_depth = process_earthquake_data(catalog)
        locs = normalize_depth(locs)
        
        split_locs = split_by_date(locs,6)
        
        lat, lon = get_geolocation(city=city)

        
        # Create a 3D plot for each split
        for i, split in enumerate(split_locs):
            
            
            start_time = split[0, 3]
            end_time = split[-1, 3]
            
            # Get Time range of each split
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            time_range_str = f"{start_time_str} to {end_time_str}"
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(split[:, 0], split[:, 1], split[:, 2] * max_depth, 
                                 label=f'Events', marker='o')
        
            ax.scatter(lon, lat, 0, c='black', marker='*', s=200, label='City')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Depth (km)')

            ax.set_title(f"""Hierarchical Clustering of Earthquake Locations for {city} (Latitude, Longitude, Depth) 
                         \nbetween {time_range_str}""")
            ax.legend()

            ax.view_init(elev=elev, azim=azim)

            plt.show()
    else:
        return None