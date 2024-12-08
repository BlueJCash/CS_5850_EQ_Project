# Small Seismic Event Analysis Project
By Thomas Prouty

## Overview
The goal of the project was to investigate the effectiveness of different visualization and prediction methods for displaying and understanding historical seismic data. This project provides tools for performing seismic analysis, visualization, clustering, and predictions for earthquake data. Specifically, the code supports hierarchical clustering, heatmap visualization, k-means clustering, and forecasting/backcasting using Random Forests and ARIMA models. 

## Table of Contents

* Features
* Dependencies
* Setup Instructions
* Usage Instructions
* Other Functions
* Data Source
* Contact Info

## Features

1. Modularity in location calling
2. Hierarchical Clustering using Ward's Method
3. K-Means Clustering
4. Choropleth Scatterplot
5. Heat Map Without Topography
6. Heat Map With Topography
7. Forecasting and Backcasting via Random Forest and ARIMA
8. Trait or Important Feature Determination via Decision Tree

## Dependencies

To execute the code please make sure the following dependencies are installed:

folium
matplotlib
geopandas
scikit-learn
seaborn
pandas
numpy
time
usgs
statsmodels

If needed, at the top of the code there is a commented section that will auto install the dependencies, if on Linux.

## Setup Instructions

Clone the repository or code files to your local machine:

git clone https://github.com/BlueJCash/CS_5850_EQ_Project/

Additionally, the following shapefiles must be in the working directory. These datasets are necessary for choropleth mapping:

ne_110m_admin_1_states_provinces.shp  
ne_110m_admin_0_countries.shp  

## Usage Instructions

Once the repository is cloned or downloaded the EQ_Analysis_Run.ipynb file can be accessed.  
After accessing this file the code can be run by running each or all cells.  
The main() function contains multiple variables that can be edited by the user to customize the analysis.
These variables are as follows:
* city # Controls focal point of the analysis
* maxradius # Controls the radius around the focal point to get data from (in degrees)
* data_starttime # Controls the period to request data for                      
* data_endtime
* freq # Determine the frequency for prediction and estimation methods (YE = year)
* elev # Controls the location of the "camera" in relation to the 3D plots
* azim
* clusters # Determines the number of clusters to be used in the clustering methods (0 means the code will autoset the amount)
* state_name # Determines the bounds to use for the chloropathy map
* country_name
* max_tree_depth # Determines the max depth of the Decision Tree Classifier
* min_samples_leaf # Determines the minimum needed samples for a leaf node in the Decision Tree

## Other Functions

The following functions can also be called by adding a new cell and providing the correct data:

* plot_3d_clusters(locs, labels, city=city, max_depth=max_depth, elev=elev, azim=azim)
* seismic_kmeans_clusters(locs, city=city, max_depth=max_depth, k=k, elev=elev, azim=azim)
* choropleth_scatterplot(locs, state_name=state_name, country_name=country_name, city=city)
* create_heatmap(locs, city=city)
* create_heatmap_topo(locs, city=city)
* predict_Seismic_RF(city, locs, duration=20, freq=freq)
* predict_Seismic_ARIMA(city, locs, duration=20, freq=freq)
* Seismic_DT_Analysis(city, locs, max_tree_depth=max_tree_depth,  min_samples_leaf=min_samples_leaf)

## Data Source

USGS Earthquake Catalog

## Contact Info

If there are any questions please message Tom Prouty via Canvas and I will work to answer them as soon as possible.

