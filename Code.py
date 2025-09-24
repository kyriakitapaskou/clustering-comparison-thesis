# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:43:42 2025

@author: kyria
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic  # For calculating distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#Dataset Seabird
data = pd.read_csv('seabird.csv')  
data.head()

# Αντιμετώπιση Missing Values
data['lat'] = data['lat'].fillna(data['lat'].mean())
data['lon'] = data['lon'].fillna(data['lon'].mean())

print("Δεδομένα μετά την αντιμετώπιση των Missing Values:")
print(data.head())


# Ανίχνευση Outliers με χρήση IQR
Q1_lat = data['lat'].quantile(0.25)
Q3_lat = data['lat'].quantile(0.75)
IQR_lat = Q3_lat - Q1_lat

Q1_lon = data['lon'].quantile(0.25)
Q3_lon = data['lon'].quantile(0.75)
IQR_lon = Q3_lon - Q1_lon

outliers_lat = data[(data['lat'] < Q1_lat - 1.5 * IQR_lat) | (data['lat'] > Q3_lat + 1.5 * IQR_lat)]
outliers_lon = data[(data['lon'] < Q1_lon - 1.5 * IQR_lon) | (data['lon'] > Q3_lon + 1.5 * IQR_lon)]

# Αντικατάσταση outliers με τον μέσο όρο
data.loc[outliers_lat.index, 'lat'] = data['lat'].mean()
data.loc[outliers_lon.index, 'lon'] = data['lon'].mean()

print("Δεδομένα μετά την αντιμετώπιση των Outliers:")
print(data.head())

# --- Ταξινόμηση βάσει χρονικής στιγμής ---
data['date_time'] = pd.to_datetime(data['date_time'])
data = data.sort_values(by=['colony2', 'date_time'])

# --- Δημιουργία επιπλέον τροχιών με βάση τα χρονικά κενά ---
data['time_group'] = data.groupby('colony2')['date_time'].diff().dt.total_seconds().gt(3600).cumsum()


plt.figure(figsize=(10, 6))
plt.scatter(data['lon'], data['lat'], s=10, alpha=0.6, c='steelblue')
plt.title('Θέσεις Seabird')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
# --- Συναρτήσεις για υπολογισμούς ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# --- Vectorization των τροχιών ---
traj_vectors = []

# Ομαδοποίηση βάσει colony2 και time_group
for (colony2, time_group), traj_data in data.groupby(['colony2', 'time_group']):
    if len(traj_data) < 2:  # Αν έχει 1 ή 0 σημεία, δεν μπορεί να υπολογιστεί τίποτα
        continue  

    speeds = []
    bearings = []

    lat_mean = traj_data['lat'].mean()
    lon_mean = traj_data['lon'].mean()

    for i in range(1, len(traj_data)):
        point1 = (traj_data.iloc[i-1]['lat'], traj_data.iloc[i-1]['lon'])
        point2 = (traj_data.iloc[i]['lat'], traj_data.iloc[i]['lon'])
      
        # Υπολογισμός απόστασης και χρόνου
        distance = geodesic(point1, point2).meters
        time_delta = (traj_data.iloc[i]['date_time'] - traj_data.iloc[i-1]['date_time']).total_seconds()

        if time_delta > 0 and distance > 0:  # Αποφυγή διαίρεσης με το μηδέν
            speed = distance / time_delta
            speeds.append(speed)

            bearing = calculate_bearing(point1[0], point1[1], point2[0], point2[1])
            bearings.append(bearing)

    # Αν υπάρχουν δεδομένα ταχύτητας, υπολογίζουμε τις τιμές τους
    if speeds:
        avg_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        speed_range = np.max(speeds) - np.min(speeds)
    else:
        avg_speed, std_speed, speed_range = 0, 0, 0  

    if bearings:
        avg_bearing = np.mean(bearings)
        std_bearing = np.std(bearings)
        bearing_range = np.max(bearings) - np.min(bearings)
    else:
        avg_bearing, std_bearing, bearing_range = 0, 0, 0  

    # Προσθήκη δεδομένων στο DataFrame
    traj_vectors.append({
        'colony2': colony2,
        'time_group': time_group,
        'lat': lat_mean,
        'lon': lon_mean,
        'avg_speed': avg_speed,
        'std_speed': std_speed,
        'speed_range': speed_range,
        'avg_bearing': avg_bearing,
        'std_bearing': std_bearing,
        'bearing_range': bearing_range
    })

# Μετατροπή σε DataFrame
traj_vectors_df = pd.DataFrame(traj_vectors)

# Εκτύπωση αποτελεσμάτων
print(f"Συνολικές τροχιές που δημιουργήθηκαν: {len(traj_vectors_df)}")
print(traj_vectors_df.head(10))

# Extract features for k-means clustering
traj_features = traj_vectors_df[['colony2','lat', 'lon','avg_speed','std_speed','speed_range','avg_bearing', 'std_bearing','bearing_range']]

# Range of cluster numbers to try
cluster_range = range(1, min(10, len(traj_features) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(traj_features)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Check for any NaN or infinite values in segment_features
print("Missing values per column:\n", traj_features.isna().sum())
print("Infinite values present:", np.isinf(traj_features).values.any())
# If any NaN or infinite values are found, drop or replace them
traj_features =traj_features.dropna()  # Drop rows with NaN (if any)

# Define clustering features
clustering_features = ['lat', 'lon','avg_speed','std_speed','speed_range','avg_bearing', 'std_bearing','bearing_range']


# Εκτέλεση K-Means με τις νέες μεταβλητές
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
traj_vectors_df['cluster'] = kmeans.fit_predict(traj_vectors_df[clustering_features])

# Κεντροειδή και labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Αριθμός συστάδων
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of iterations: %d' % kmeans.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(traj_vectors_df[clustering_features], labels))
print("Mean Squared Error: %0.3f" % kmeans.inertia_)

# Γράφημα clustering
plt.figure(figsize=(10, 5))
plt.scatter(traj_vectors_df['avg_bearing'], traj_vectors_df['speed_range'], 
            c=traj_vectors_df['cluster'], cmap='viridis', marker='o', s=100, edgecolor='k')

# Προσθήκη κεντροειδών
plt.scatter(centroids[:, clustering_features.index('avg_bearing')], 
            centroids[:, clustering_features.index('speed_range')], 
            c='red', marker='X', s=200, label='Centroids')

plt.xlabel('avg_bearing')
plt.ylabel('speed_range')
plt.title('Cluster Separation using Coordinates and Bearing')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
traj_vectors_df['cluster'] = kmeans.fit_predict(traj_vectors_df[clustering_features])

# Κεντροειδή και labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Αριθμός συστάδων
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of iterations: %d' % kmeans.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(traj_vectors_df[clustering_features], labels))
print("Mean Squared Error: %0.3f" % kmeans.inertia_)

# Γράφημα clustering
plt.figure(figsize=(10, 5))
plt.scatter(traj_vectors_df['avg_bearing'], traj_vectors_df['speed_range'], 
            c=traj_vectors_df['cluster'], cmap='viridis', marker='o', s=100, edgecolor='k')

# Προσθήκη κεντροειδών
plt.scatter(centroids[:, clustering_features.index('avg_bearing')], 
            centroids[:, clustering_features.index('speed_range')], 
            c='red', marker='X', s=200, label='Centroids')

plt.xlabel('avg_bearing')
plt.ylabel('speed_range')
plt.title('Cluster Separation using Coordinates and Bearing')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


print(traj_vectors_df['cluster'].value_counts())

# Define clustering features
clustering_features = ['lat', 'lon','avg_speed','std_speed','speed_range','avg_bearing', 'std_bearing','bearing_range']

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(
    traj_vectors_df['lon'], 
    traj_vectors_df['lat'],
    c=traj_vectors_df['cluster'],      # Χρώμα ανά cluster
    cmap='tab10',
    s=30,
    alpha=0.7,
    edgecolor='k'
)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters στον Γεωγραφικό Χώρο')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Create list to store results
results = []

# Range of k values
k_range = range(2, 11)

# Subplot grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Flatten into 1D array for easier indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(traj_vectors_df[clustering_features])
    
    # Calculate Silhouette Coefficient
    silhouette = silhouette_score(traj_vectors_df[clustering_features], labels)
    
    # Calculate Mean Squared Error (SSE)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualization of clusters
    ax = axes[idx]
    ax.scatter(traj_vectors_df['avg_bearing'], 
               traj_vectors_df['speed_range'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, clustering_features.index('avg_bearing')], 
               centroids[:, clustering_features.index('speed_range')], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('avg_bearing')
    ax.set_ylabel('speed_range')
    ax.legend()
    ax.grid(True)

# Hide unused subplot axes
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional Visualization of Metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()


# Define different parameter scenarios for OPTICS clustering and DBSCAN eps values
scenarios = [
    {'min_samples': 3, 'xi': 0.02, 'min_cluster_size': 0.05, 'eps_1': 4, 'eps_2': 6},
    {'min_samples': 4, 'xi': 0.04, 'min_cluster_size': 0.07, 'eps_1': 3, 'eps_2': 6},
    {'min_samples': 5, 'xi': 0.03, 'min_cluster_size': 0.1, 'eps_1': 5, 'eps_2': 7},
    {'min_samples': 6, 'xi': 0.02, 'min_cluster_size': 0.1, 'eps_1': 6, 'eps_2': 8},
    {'min_samples': 5, 'xi': 0.05, 'min_cluster_size': 0.12, 'eps_1': 4, 'eps_2': 7},
]


def run_optics_scenarios(scenarios,data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['lat', 'lon', 'avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        silhouette_avg_1 = silhouette_score(data[['lat', 'lon', 'avg_bearing', 'std_bearing']], labels_100)
        silhouette_avg_2 = silhouette_score(data[['lat', 'lon', 'avg_bearing', 'std_bearing']], labels_120)
        # Calculate silhouette scores only if there are at least 2 clusters
        
        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results

def run_optics_scenarios(scenarios, data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['lat', 'lon', 'avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        # Ensure silhouette scores are calculated only for valid clusterings
        if len(set(labels_100)) > 1:
            silhouette_avg_1 = silhouette_score(data[['lat', 'lon', 'avg_bearing', 'std_bearing']], labels_100)
        else:
            silhouette_avg_1 = -1  # Invalid clustering (only one cluster)
        
        if len(set(labels_120)) > 1:
            silhouette_avg_2 = silhouette_score(data[['lat', 'lon', 'avg_bearing', 'std_bearing']], labels_120)
        else:
            silhouette_avg_2 = -1  # Invalid clustering (only one cluster)

        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results
results = run_optics_scenarios(scenarios, traj_vectors_df)
for res in results:
    print(res)
    
    results = run_optics_scenarios(scenarios, traj_vectors_df)

# Find the best result based on silhouette_avg_1 and silhouette_avg_2
best_result_1 = max(results, key=lambda x: x['silhouette_avg_1'])
best_result_2 = max(results, key=lambda x: x['silhouette_avg_2'])

# Print the best results
print("Best Result for eps_1:")
print(f"Scenario: {best_result_1['scenario']}")
print(f"Parameters: {best_result_1['params']}")
print(f"Silhouette Score: {best_result_1['silhouette_avg_1']}")

print("\nBest Result for eps_2:")
print(f"Scenario: {best_result_2['scenario']}")
print(f"Parameters: {best_result_2['params']}")
print(f"Silhouette Score: {best_result_2['silhouette_avg_2']}")

# Εκπαίδευση OPTICS
clust = OPTICS(min_samples=6, metric='euclidean', xi=0.02, min_cluster_size=0.1)
clust.fit(traj_vectors_df[['lat', 'lon', 'avg_speed', 'std_speed', 'speed_range',
                           'avg_bearing', 'std_bearing', 'bearing_range']])

# DBSCAN "cuts"
labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=6)
labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=8)

# Reachability plot preparation
space = np.arange(len(traj_vectors_df))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
valid_reachability = reachability[np.isfinite(reachability)]

# Latitude / Longitude bounds
lat_min, lat_max = traj_vectors_df['lat'].min(), traj_vectors_df['lat'].max()
lon_min, lon_max = traj_vectors_df['lon'].min(), traj_vectors_df['lon'].max()

# Set up plot
plt.figure(figsize=(15, 10))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])  # Reachability Plot
ax2 = plt.subplot(G[1, 0])  # OPTICS
ax3 = plt.subplot(G[1, 1])  # DBSCAN eps=6
ax4 = plt.subplot(G[1, 2])  # DBSCAN eps=8

# === Reachability Plot ===
ax1.set_xlim(0, len(space))
ax1.set_ylim(0, valid_reachability.max() + 10 if len(valid_reachability) > 0 else 10)
ax1.set_title('Reachability Plot')
ax1.set_ylabel('Reachability (epsilon distance)')

# Χρώματα: έως 5 clusters
colors = ['g.', 'r.', 'y.', 'b.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.6)

# Θόρυβος (noise)
Xk_noise = space[labels == -1]
Rk_noise = reachability[labels == -1]
ax1.plot(Xk_noise, Rk_noise, 'k+', alpha=0.3)

# Οδηγοί για eps
ax1.plot(space, np.full_like(space, 6.0), 'k--', alpha=0.4, label='eps=6')
ax1.plot(space, np.full_like(space, 2.0), 'k:', alpha=0.4, label='eps=2')
ax1.legend()

# === Συνάρτηση για τα cluster plots ===
def plot_clusters(ax, labels, title):
    colors = ['g.', 'r.', 'y.', 'b.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = traj_vectors_df[labels == klass]
        ax.plot(Xk['lon'], Xk['lat'], color, alpha=0.8)
    # Noise
    Xk_noise = traj_vectors_df[labels == -1]
    ax.plot(Xk_noise['lon'], Xk_noise['lat'], 'k+', alpha=0.3)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

# === Τρισδιάστατα cluster plots ===
plot_clusters(ax2, clust.labels_, "Automatic Clustering\nOPTICS")
plot_clusters(ax3, labels_100, "DBSCAN Clustering\n(eps=6)")
plot_clusters(ax4, labels_120, "DBSCAN Clustering\n(eps=8)")

plt.tight_layout()
plt.show()
--------------------------------------------------------------------------------------------------

#Methodos Pca Seabird

data = pd.read_csv('seabird.csv')
data['lat'] = data['lat'].fillna(data['lat'].mean())
data['lon'] = data['lon'].fillna(data['lon'].mean())

# Ανίχνευση Outliers με IQR
Q1_lat, Q3_lat = data['lat'].quantile([0.25, 0.75])
IQR_lat = Q3_lat - Q1_lat
Q1_lon, Q3_lon = data['lon'].quantile([0.25, 0.75])
IQR_lon = Q3_lon - Q1_lon

outliers_lat = data[(data['lat'] < Q1_lat - 1.5 * IQR_lat) | (data['lat'] > Q3_lat + 1.5 * IQR_lat)]
outliers_lon = data[(data['lon'] < Q1_lon - 1.5 * IQR_lon) | (data['lon'] > Q3_lon + 1.5 * IQR_lon)]
data.loc[outliers_lat.index, 'lat'] = data['lat'].mean()
data.loc[outliers_lon.index, 'lon'] = data['lon'].mean()

# Ταξινόμηση βάσει ημερομηνίας
data['date_time'] = pd.to_datetime(data['date_time'])
data = data.sort_values(by=['colony2', 'date_time'])

# --- Interpolation function ---
def interpolate_trajectory(lat, lon, t, n_points=100):
    if len(lat) < 2:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)

    # Convert to pandas datetime and calculate seconds
    t = pd.to_datetime(t)
    t_seconds = (t - t.min()).dt.total_seconds()

    interpolated_times = np.linspace(t_seconds.min(), t_seconds.max(), n_points)
    lat_interpolator = interp1d(t_seconds, lat, kind='linear', fill_value="extrapolate")
    lon_interpolator = interp1d(t_seconds, lon, kind='linear', fill_value="extrapolate")

    interpolated_lat = lat_interpolator(interpolated_times)
    interpolated_lon = lon_interpolator(interpolated_times)
    
    return interpolated_lat, interpolated_lon

# --- Επιλογή τροχιάς ---
desired_traj_id = 1  # ή άλλο ID που θες
single_traj_data = data[data['colony2'] == desired_traj_id]

# Επιλέγουμε ένα λογικό υποσύνολο (π.χ. 100 συνεχόμενα σημεία)
single_traj_data = single_traj_data.iloc[100:200]

# Παίρνουμε τις στήλες
lat = single_traj_data['lat'].values
lon = single_traj_data['lon'].values
t = single_traj_data['date_time']

# Interpolation
n_points = 100
interpolated_lat, interpolated_lon = interpolate_trajectory(lat, lon, t, n_points)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(lon, lat, 'o-', label='Original Trajectory', alpha=0.6)
plt.plot(interpolated_lon, interpolated_lat, 'r-', label='Interpolated Trajectory', linewidth=2)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title(f'Trajectory Interpolation for colony2 = {desired_traj_id}')
plt.legend()
plt.grid()
plt.show()


# Apply to each trajectory
n_interpolation_points = 100 # Choose a fixed number of points for representation
trajectory_features = []

for colony2, single_traj_data in data.groupby('colony2'):
    c1s = single_traj_data['lat'].values
    c2s = single_traj_data['lon'].values
    ts = single_traj_data['date_time'] 
    
    # Interpolate the trajectory
    interp_c1, interp_c2 = interpolate_trajectory(c1s,c2s,ts, n_points=n_interpolation_points)
    
    # Combine latitude and longitude features
    trajectory_vector = np.concatenate([interp_c1, interp_c2])
    trajectory_features.append({
        'colony2': colony2,
        'trajectory_vector': trajectory_vector
    })

# Convert to DataFrame
trajectory_features_df = pd.DataFrame(trajectory_features)
trajectory_features_df.info()
X = np.vstack(trajectory_features_df['trajectory_vector'].values)

# Standardize the features (important for PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Check the explained variance ratio for each number of PCA components
n_components = 10  # Test the first 10 components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Print the explained variance and cumulative variance
print("PCA Component-wise Explained Variance and Cumulative Variance:")
for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"Component {i+1}: Explained Variance = {var_ratio:.4f}, Cumulative Variance = {cum_var:.4f}")

# Step 2: Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.show()

# Step 3: Find the optimal number of components using clustering (e.g., KMeans with Silhouette Score)
silhouette_scores = []

print("\nSilhouette Scores for KMeans Clustering:")
for n in range(1, n_components + 1):
    pca = PCA(n_components=n)
    X_reduced = pca.fit_transform(X_scaled)

    # Apply KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)

    # Calculate silhouette score
    sil_score = silhouette_score(X_reduced, clusters)
    silhouette_scores.append(sil_score)
    print(f"PCA Components: {n}, Silhouette Score: {sil_score:.4f}")

# Step 4: Plot silhouette scores to determine optimal components
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), silhouette_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Number of PCA Components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of PCA Components')
plt.grid()
plt.show()

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Προσθήκη των πρώτων 38 συνιστωσών στο DataFrame (δυναμικά)
for i in range(2):
    trajectory_features_df[f'pca_{i + 1}'] = X_reduced[:, i]

# Εμφάνιση του DataFrame με τις πρώτες δύο συνιστώσες
print(trajectory_features_df[['colony2', 'pca_1', 'pca_2']].head())

# Κανονικοποίηση του X_reduced 
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_reduced)  

# Εκτύπωση πληροφοριών για έλεγχο
print("Shape of reduced data:", X_reduced.shape)
print("Shape of normalized data:", X_normalized.shape)

print(trajectory_features_df)

# Range of cluster numbers to try
cluster_range = range(1, min(10, len(X_normalized) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=30)
    kmeans.fit(X_normalized)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Run K-Means with a predefined number of clusters
n_clusters =3 # Adjust this based on your dataset and goal
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
clusters = kmeans.fit_predict(X_normalized)

# Add the cluster labels back to the trajectory features DataFrame
trajectory_features_df['cluster'] = clusters

# Inspect the clustering results
print(trajectory_features_df[['colony2', 'cluster']].head())

# Calculate silhouette score
sil_score = silhouette_score(X_normalized, clusters)
print(f"Silhouette Score: {sil_score}")

# Visualize the clusters in 2D using PCA components
plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    cluster_points = trajectory_features_df[trajectory_features_df['cluster'] == cluster_id]
    plt.scatter(cluster_points['pca_1'], cluster_points['pca_2'], label=f"Cluster {cluster_id}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("K-Means Clusters of Trajectories")
plt.show()

# Create a list to store results
results = []

# Range of k values to test
k_range = range(2, 11)

# Size of subplots for visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Convert to 1D list for easy indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = kmeans.fit_predict(X_normalized)  # Use the normalized trajectory vectors
    
    # Calculate Silhouette Coefficient (only if there are at least 2 clusters)
    silhouette = silhouette_score(X_normalized, labels)
    
    # Calculate Sum of Squared Errors (SSE or Inertia)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualize the clusters in PCA space (use PCA components for visualization)
    ax = axes[idx]
    ax.scatter(trajectory_features_df['pca_1'], 
               trajectory_features_df['pca_2'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids (transformed to PCA space)
    centroids = kmeans.cluster_centers_
    centroids_pca = PCA(n_components=2).fit_transform(centroids)  # Transform to PCA space
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    ax.grid(True)

# Hide empty plots if there are fewer than 9 clusters
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional visualization of metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error (SSE)
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()

#DBSCAN
# Step 1: Scale the data and reduce dimensionality using PCA (already done above)
# Ensure X_pca_reduced is the PCA-reduced data (e.g., 2 components)

# Step 2: Run DBSCAN with varying eps and min_samples
eps_values = np.arange(0.1, 3.0, 0.1)  # Adjust the range based on dataset scale
min_samples_values = [2, 3, 4]  # Different density thresholds
best_eps = None
best_silhouette = -1
best_dbscan = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_normalized)

        # Skip silhouette score calculation if all points are noise or in one cluster
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_normalized, labels)
            print(f"DBSCAN: eps={eps}, min_samples={min_samples}, silhouette={silhouette}")
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_dbscan = dbscan
                best_labels = labels

print(f"Best DBSCAN: eps={best_eps}, silhouette={best_silhouette}")

if best_labels is None:
    print("⚠️ No valid clustering configuration found (all noise or one cluster each time).")
else:
    # Step 3: Visualize the clusters from the best DBSCAN model
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        if label == -1:  # Noise points
            plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
        else:
            plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Cluster {label}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"DBSCAN Clustering (eps={best_eps})")
    plt.legend()
    plt.show()

#try2
# Initialize OPTICS
min_samples_values = [2, 3, 4,5]  # Varying min_samples values
best_silhouette = -1
best_min_samples = None
best_labels = None

# Loop over min_samples to test different configurations
for min_samples in min_samples_values:
    optics = OPTICS(min_samples=min_samples, metric='minkowski',xi=0.03, min_cluster_size=0.3)
    labels = optics.fit_predict(X_normalized)  # X_normalized: your scaled trajectory vectors
    
    # Skip silhouette score calculation if all points are noise or in one cluster
    if len(set(labels)) > 1:  # Ensure more than one cluster exists
        silhouette = silhouette_score(X_normalized, labels)
        print(f"OPTICS: min_samples={min_samples}, silhouette={silhouette}")
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_min_samples = min_samples
            best_labels = labels

if best_labels is not None:
    print(f"Best OPTICS: min_samples={best_min_samples}, silhouette={best_silhouette}")

    # Visualization of OPTICS results
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        if label == -1:  # Noise points
            plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
        else:
            plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Cluster {label}')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"OPTICS Clustering (min_samples={best_min_samples})")
    plt.legend()
    plt.show()
else:
    print("OPTICS could not find any meaningful clusters with the given parameters.")

# Optional: Extract Reachability Plot
reachability = optics.reachability_[optics.ordering_]
plt.figure(figsize=(10, 6))
plt.plot(reachability)
plt.title("OPTICS Reachability Plot")
plt.xlabel("Sample Index")
plt.ylabel("Reachability Distance")
plt.show()

-------------------------------------------------------------------------------------------------

#Dataset Animals
data = pd.read_csv('animals_preapred.txt')  
data.head()

# Αντιμετώπιση Missing Values
data['c1'] = data['c1'].fillna(data['c1'].mean())
data['c2'] = data['c2'].fillna(data['c2'].mean())

print("Δεδομένα μετά την αντιμετώπιση των Missing Values:")
print(data.head())

# Ανίχνευση Outliers με χρήση IQR
Q1_c1 = data['c1'].quantile(0.25)
Q3_c1 = data['c1'].quantile(0.75)
IQR_c1 = Q3_c1 - Q1_c1

Q1_c2 = data['c2'].quantile(0.25)
Q3_c2 = data['c2'].quantile(0.75)
IQR_c2 = Q3_c2 - Q1_c2

outliers_c1 = data[(data['c1'] < Q1_c1 - 1.5 * IQR_c1) | (data['c1'] > Q3_c1 + 1.5 * IQR_c1)]
outliers_c2 = data[(data['c2'] < Q1_c2 - 1.5 * IQR_c2) | (data['c2'] > Q3_c2 + 1.5 * IQR_c2)]

# Αντικατάσταση outliers με τον μέσο όρο
data.loc[outliers_c1.index, 'c1'] = data['c1'].mean()
data.loc[outliers_c2.index, 'c2'] = data['c2'].mean()

print("Δεδομένα μετά την αντιμετώπιση των Outliers:")
print(data.head())

# Ταξινόμηση των δεδομένων βάσει χρονικής στιγμής (t)
data['t'] = pd.to_datetime(data['t'])
data = data.sort_values(by='t')

plt.figure(figsize=(10, 6))
plt.scatter(data['c1'], data['c2'], s=10, alpha=0.6, c='steelblue')
plt.title('Θέσεις Animals')
plt.xlabel('c1')
plt.ylabel('c2')
plt.grid(True)
plt.show()

#Vectorisation
# Function to calculate bearing (direction) between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# calculate for each id
traj_vectors = []

# Group data by trajectory
for tid, traj_data in data.groupby('tid'):
    speeds = []
    bearings = []
    
    c1_mean = traj_data['c1'].mean()
    c2_mean = traj_data['c2'].mean()
    
    for i in range(1, len(traj_data)):
        point1 = (traj_data.iloc[i-1]['c2'], traj_data.iloc[i-1]['c1'])
        point2 = (traj_data.iloc[i]['c2'], traj_data.iloc[i]['c1'])
        
        # calculate distance and time
        distance = geodesic(point1, point2).meters
        time_delta = (traj_data.iloc[i]['t'] - traj_data.iloc[i-1]['t']).total_seconds()
        
        # calculate speed and bearing  
        speed = distance / time_delta if time_delta > 0 else 0
        speeds.append(speed)
        
        bearing = calculate_bearing(point1[0], point1[1], point2[0], point2[1])
        bearings.append(bearing)
    
    # Calculation of average speed and bearing for the flight
    avg_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    speed_range = np.max(speeds) - np.min(speeds)
   
    avg_bearing = np.mean(bearings)
    std_bearing = np.std(bearings)
    bearing_range = np.max(bearings) - np.min(bearings)
    
    # Adding the flight vector to the list
    traj_vectors.append({
        'tid': tid,
        'c1': c1_mean,        # Μέση τιμή c1
        'c2': c2_mean,        # Μέση τιμή c2
        'avg_speed': avg_speed,
        'std_speed': std_speed,
        'speed_range': speed_range,
        'avg_bearing': avg_bearing,
        'std_bearing': std_bearing,
        'bearing_range': bearing_range
    })

# Convert to DataFrame for easier viewing
traj_vectors_df = pd.DataFrame(traj_vectors)
print(len(traj_vectors))

# Extract features for k-means clustering
traj_features = traj_vectors_df[['tid','c1', 'c2','avg_speed','std_speed','speed_range','avg_bearing', 'std_bearing','bearing_range']]


# Range of cluster numbers to try
cluster_range = range(1, min(10, len(traj_features) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(traj_features)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Check for any NaN or infinite values in segment_features
print("Missing values per column:\n", traj_features.isna().sum())
print("Infinite values present:", np.isinf(traj_features).values.any())
# If any NaN or infinite values are found, drop or replace them
traj_features =traj_features.dropna()  # Drop rows with NaN (if any)
print(traj_vectors_df.columns)

# Νέα χαρακτηριστικά για clustering
clustering_features = ['c1', 'c2', 'avg_bearing', 'std_bearing']

# Εκτέλεση K-Means με τις νέες μεταβλητές
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
traj_vectors_df['cluster'] = traj_vectors_df['cluster'].astype(int)

# Ορισμός χρωμάτων για τα clusters
cluster_colors = ['purple', 'green']

plt.figure(figsize=(10, 5))

# Σχεδίαση κάθε cluster ξεχωριστά για έλεγχο του χρώματος
for i in range(optimal_k):
    cluster_points = traj_vectors_df[traj_vectors_df['cluster'] == i]
    plt.scatter(cluster_points['c1'], cluster_points['c2'],
                color=cluster_colors[i], label=f'Cluster {i}', s=100, edgecolor='k')

# Προσθήκη κεντροειδών
plt.scatter(centroids[:, clustering_features.index('c1')], 
            centroids[:, clustering_features.index('c2')],
            c='red', marker='X', s=200, label='Centroids')

plt.xlabel('c1')
plt.ylabel('c2')
plt.title('Cluster Separation using Coordinates and Bearing')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Βεβαιώσου ότι το 'cluster' είναι ακέραιος τύπος
traj_vectors_df['cluster'] = traj_vectors_df['cluster'].astype(int)

# Έλεγχος ότι έχεις 2 διαφορετικά clusters
print(traj_vectors_df['cluster'].value_counts())

# Χρώματα για τα δύο clusters
colors = {0: 'purple', 1: 'green'}

plt.figure(figsize=(12, 6))

# Σχεδίαση κάθε cluster ξεχωριστά
for cluster_id in traj_vectors_df['cluster'].unique():
    cluster_points = traj_vectors_df[traj_vectors_df['cluster'] == cluster_id]
    plt.scatter(cluster_points['c1'], cluster_points['c2'],
                c=colors[cluster_id], label=f'Cluster {cluster_id}',
                s=100, edgecolor='k')

# Σχεδίαση κεντροειδών
plt.scatter(centroids[:, clustering_features.index('c1')],
            centroids[:, clustering_features.index('c2')],
            c='red', marker='X', s=200, label='Centroids')

plt.xlabel('c1')
plt.ylabel('c2')
plt.title('Cluster Separation using Coordinates and Bearing')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# KMeans 
# k=4
clustering_features = ['c1', 'c2', 'avg_bearing', 'std_bearing']

# Εκτέλεση K-Means με τις νέες μεταβλητές
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
traj_vectors_df['cluster'] = kmeans.fit_predict(traj_vectors_df[clustering_features])

# Κεντροειδή και labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Αριθμός συστάδων
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of iterations: %d' % kmeans.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(traj_vectors_df[clustering_features], labels))
print("Mean Squared Error: %0.3f" % kmeans.inertia_)

# Γράφημα clustering
plt.figure(figsize=(10, 5))
plt.scatter(traj_vectors_df['c1'], traj_vectors_df['c2'], 
            c=traj_vectors_df['cluster'], cmap='viridis', marker='o', s=100, edgecolor='k')

# Προσθήκη κεντροειδών
plt.scatter(centroids[:, clustering_features.index('c1')], 
            centroids[:, clustering_features.index('c2')], 
            c='red', marker='X', s=200, label='Centroids')

plt.xlabel('c1')
plt.ylabel('c2')
plt.title('Cluster Separation using Coordinates and Bearing')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

print(traj_vectors_df['cluster'].value_counts())


# Define clustering features
clustering_features = ['c1', 'c2', 'avg_bearing', 'std_bearing']

plt.figure(figsize=(10, 6))
plt.scatter(
    traj_vectors_df['c1'], 
    traj_vectors_df['c2'],
    c=traj_vectors_df['cluster'],      # Χρώμα ανά cluster
    cmap='tab10',
    s=30,
    alpha=0.7,
    edgecolor='k'
)
plt.xlabel('c1')
plt.ylabel('c2')
plt.title('Clusters στον Γεωγραφικό Χώρο')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Create list to store results
results = []

# Range of k values
k_range = range(2, 11)

# Subplot grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Flatten into 1D array for easier indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(traj_vectors_df[clustering_features])
    
    # Calculate Silhouette Coefficient
    silhouette = silhouette_score(traj_vectors_df[clustering_features], labels)
    
    # Calculate Mean Squared Error (SSE)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualization of clusters
    ax = axes[idx]
    ax.scatter(traj_vectors_df['c1'], 
               traj_vectors_df['c2'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, clustering_features.index('c1')], 
               centroids[:, clustering_features.index('c2')], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('c1')
    ax.set_ylabel('c2')
    ax.legend()
    ax.grid(True)

# Hide unused subplot axes
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional Visualization of Metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()


# OPTICS Clustering
clust = OPTICS(min_samples=15, metric='euclidean', xi=0.01, min_cluster_size=0.2)

# Apply OPTICS on the selected features
clust.fit(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# Generate DBSCAN labels at different epsilon values
labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=1)
labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=5)

# Prepare data for the reachability plot
space = np.arange(len(traj_vectors_df))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

# Clean reachability array to remove NaN or Inf values
valid_reachability = reachability[np.isfinite(reachability)]

# Plotting
plt.figure(figsize=(15, 10))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Set axis limits dynamically (avoid NaN or Inf errors)
if len(valid_reachability) > 0:  # Ensure there are valid values
    ax1.set_ylim(0, valid_reachability.max() + 10)
else:
    ax1.set_ylim(0, 10)  # Fallback y-axis limit

ax1.set_xlim(0, len(space))

# Set static limits for clustering plots
ax2.set_xlim(0, 1000)
ax2.set_ylim(100, 300)
ax3.set_xlim(0, 1000)
ax3.set_ylim(100, 300)
ax4.set_xlim(0, 1000)
ax4.set_ylim(100, 300)

# Reachability Plot
colors = ['g.', 'r.', 'y.', 'b.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)

# Plot noise points
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS Automatic Clustering
for klass, color in zip(range(0, 5), colors):
    Xk = traj_vectors_df[clust.labels_ == klass]
    ax2.plot(Xk['c1'], Xk['c2'], color, alpha=1)
ax2.plot(traj_vectors_df[clust.labels_ == -1]['c1'], 
         traj_vectors_df[clust.labels_ == -1]['c2'], 'k.', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN Clustering with eps=23
for klass, color in zip(range(0, 5), colors):
    Xk = traj_vectors_df[labels_100 == klass]
    ax3.plot(Xk['c1'], Xk['c2'], color, alpha=0.3)
ax3.plot(traj_vectors_df[labels_100 == -1]['c1'], 
         traj_vectors_df[labels_100 == -1]['c2'], 'k+', alpha=0.1)
ax3.set_title('Clustering at 1 epsilon cut\nDBSCAN')

# DBSCAN Clustering with eps=26
for klass, color in zip(range(0, 5), colors):
    Xk = traj_vectors_df[labels_120 == klass]
    ax4.plot(Xk['c1'], Xk['c2'], color, alpha=0.3)
ax4.plot(traj_vectors_df[labels_120 == -1]['c1'], 
         traj_vectors_df[labels_120 == -1]['c2'], 'k+', alpha=0.1)
ax4.set_title('Clustering at 5 epsilon cut\nDBSCAN')

# Finalize Layout
plt.tight_layout()
plt.show()

# Λίστες για αποθήκευση αποτελεσμάτων
results_optics = []

# Υπολογισμός Silhouette και SSE για OPTICS
for epsilon, labels in zip([1, 5], [labels_100, labels_120]):
    # Εξαιρούμε σημεία με θόρυβο (label = -1)
    valid_points = traj_vectors_df[labels != -1]
    valid_labels = labels[labels != -1]
    
    if len(np.unique(valid_labels)) > 1:  # Υπολογίζουμε Silhouette μόνο αν έχουμε >1 cluster
        silhouette = silhouette_score(valid_points[[ 'c1', 'c2', 'avg_bearing', 'std_bearing']], valid_labels)
    else:
        silhouette = np.nan  # Δεν ορίζεται για ένα μόνο cluster
    
    # Υπολογισμός SSE (άθροισμα αποστάσεων από τα "κεντροειδή" κάθε cluster)
    sse = 0
    for cluster in np.unique(valid_labels):
        cluster_points = valid_points[valid_labels == cluster]
        centroid = cluster_points[[ 'c1', 'c2', 'avg_bearing', 'std_bearing']].mean(axis=0)
        sse += np.sum((cluster_points[[ 'c1', 'c2', 'avg_bearing', 'std_bearing']] - centroid) ** 2)
    
    # Αποθήκευση αποτελεσμάτων
    results_optics.append({
        'Epsilon': epsilon,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': sse
    })

# Μετατροπή σε DataFrame για καλύτερη εμφάνιση
results_optics_df = pd.DataFrame(results_optics)

# Ρύθμιση για εμφάνιση όλων των στηλών χωρίς περικοπή
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Εμφάνιση αποτελεσμάτων
print(results_optics_df)

# Define different parameter scenarios for OPTICS clustering and DBSCAN eps values
scenarios = [
    {'min_samples': 5, 'xi': 0.02, 'min_cluster_size': 0.1, 'eps_1': 3, 'eps_2': 5},
    {'min_samples': 5, 'xi': 0.03, 'min_cluster_size': 0.1, 'eps_1': 5, 'eps_2': 7},
    {'min_samples': 3, 'xi': 0.02, 'min_cluster_size': 0.1, 'eps_1': 3, 'eps_2': 5},
    {'min_samples': 3, 'xi': 0.03, 'min_cluster_size': 0.1, 'eps_1': 5, 'eps_2': 7},
    {'min_samples': 5, 'xi': 0.02, 'min_cluster_size': 0.15, 'eps_1': 2, 'eps_2': 5},
    {'min_samples': 5, 'xi': 0.05, 'min_cluster_size': 0.15, 'eps_1': 3, 'eps_2': 6},
]


def run_optics_scenarios(scenarios,data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['c1', 'c2', 'avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        silhouette_avg_1 = silhouette_score(data[['c1', 'c2', 'avg_bearing', 'std_bearing']], labels_100)
        silhouette_avg_2 = silhouette_score(data[['c1', 'c2', 'avg_bearing', 'std_bearing']], labels_120)
        # Calculate silhouette scores only if there are at least 2 clusters
        
        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results

def run_optics_scenarios(scenarios, data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['c1', 'c2', 'avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        # Ensure silhouette scores are calculated only for valid clusterings
        if len(set(labels_100)) > 1:
            silhouette_avg_1 = silhouette_score(data[['c1', 'c2', 'avg_bearing', 'std_bearing']], labels_100)
        else:
            silhouette_avg_1 = -1  # Invalid clustering (only one cluster)
        
        if len(set(labels_120)) > 1:
            silhouette_avg_2 = silhouette_score(data[['c1', 'c2', 'avg_bearing', 'std_bearing']], labels_120)
        else:
            silhouette_avg_2 = -1  # Invalid clustering (only one cluster)

        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results
results = run_optics_scenarios(scenarios, traj_vectors_df)
for res in results:
    print(res)
    
    results = run_optics_scenarios(scenarios, traj_vectors_df)

# Find the best result based on silhouette_avg_1 and silhouette_avg_2
best_result_1 = max(results, key=lambda x: x['silhouette_avg_1'])
best_result_2 = max(results, key=lambda x: x['silhouette_avg_2'])

# Print the best results
print("Best Result for eps_1:")
print(f"Scenario: {best_result_1['scenario']}")
print(f"Parameters: {best_result_1['params']}")
print(f"Silhouette Score: {best_result_1['silhouette_avg_1']}")

print("\nBest Result for eps_2:")
print(f"Scenario: {best_result_2['scenario']}")
print(f"Parameters: {best_result_2['params']}")
print(f"Silhouette Score: {best_result_2['silhouette_avg_2']}")

# Reachability Plot
def plot_reachability(clust):
    space = np.arange(len(clust.reachability_))
    reachability = clust.reachability_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    plt.plot(space, reachability, 'b.', alpha=0.5)
    plt.title("Reachability Plot (OPTICS)")
    plt.xlabel("Sample Index")
    plt.ylabel("Reachability Distance")
    plt.show()

# DBSCAN Plot
def plot_dbscan(data, labels):
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Θόρυβος (noise points)
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask]

        plt.plot(xy['c1'], xy['c2'], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title("DBSCAN Clustering")
    plt.xlabel("c1")
    plt.ylabel("c2")
    plt.show()

# Εκπαίδευση OPTICS
clust = OPTICS(min_samples=5, metric='euclidean', xi=0.03, min_cluster_size=0.1)
clust.fit(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# Reachability Plot
plot_reachability(clust)

# DBSCAN Clustering με τα ίδια δεδομένα
labels = DBSCAN(eps=5, min_samples=5).fit_predict(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# DBSCAN Plot
plot_dbscan(traj_vectors_df, labels)


# Εκπαίδευση OPTICS
clust = OPTICS(min_samples=5, metric='euclidean', xi=0.03, min_cluster_size=0.1)
clust.fit(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# Reachability Plot
plot_reachability(clust)

# DBSCAN Clustering με τα ίδια δεδομένα
labels = DBSCAN(eps=7, min_samples=5).fit_predict(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# DBSCAN Plot
plot_dbscan(traj_vectors_df, labels)

# Συνάρτηση για Reachability Plot (OPTICS)
def plot_reachability(ax, clust, title):
    space = np.arange(len(clust.reachability_))
    reachability = clust.reachability_[clust.ordering_]
    ax.plot(space, reachability, 'b.', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Reachability Distance")

# Συνάρτηση για DBSCAN Plot
def plot_dbscan(ax, data, labels, title):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Θόρυβος (noise points)
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask]

        ax.plot(xy['c1'], xy['c2'], 'o', markerfacecolor=tuple(col), 
                markeredgecolor='k', markersize=6)

    ax.set_title(title)
    ax.set_xlabel("c1")
    ax.set_ylabel("c2")

# Εκπαίδευση OPTICS
clust = OPTICS(min_samples=5, metric='euclidean', xi=0.03, min_cluster_size=0.1)
clust.fit(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# DBSCAN Clustering με eps=3 και eps=5
labels_eps3 = DBSCAN(eps=5, min_samples=5).fit_predict(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])
labels_eps5 = DBSCAN(eps=7, min_samples=5).fit_predict(traj_vectors_df[['c1', 'c2', 'avg_bearing', 'std_bearing']])

# Δημιουργία σχήματος με 4 υποπλοκές (subplots)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Reachability Plots
plot_reachability(axs[0, 0], clust, "Reachability Plot (OPTICS, eps=5)")
plot_reachability(axs[0, 1], clust, "Reachability Plot (OPTICS, eps=7)")

# DBSCAN Plots
plot_dbscan(axs[1, 0], traj_vectors_df, labels_eps3, "DBSCAN Clustering (eps=5)")
plot_dbscan(axs[1, 1], traj_vectors_df, labels_eps5, "DBSCAN Clustering (eps=7)")

# Κενά για καλύτερη εμφάνιση
plt.tight_layout()
plt.show()

--------------------------------------------------------------------------------------------------------
#Methodos PCA Animals
data = pd.read_csv('animals_preapred.txt')  
data.head()

# Αντιμετώπιση Missing Values
data['c1'] = data['c1'].fillna(data['c1'].mean())
data['c2'] = data['c2'].fillna(data['c2'].mean())

print("Δεδομένα μετά την αντιμετώπιση των Missing Values:")
print(data.head())

# Ανίχνευση Outliers με χρήση IQR
Q1_c1 = data['c1'].quantile(0.25)
Q3_c1 = data['c1'].quantile(0.75)
IQR_c1 = Q3_c1 - Q1_c1

Q1_c2 = data['c2'].quantile(0.25)
Q3_c2 = data['c2'].quantile(0.75)
IQR_c2 = Q3_c2 - Q1_c2

outliers_c1 = data[(data['c1'] < Q1_c1 - 1.5 * IQR_c1) | (data['c1'] > Q3_c1 + 1.5 * IQR_c1)]
outliers_c2 = data[(data['c2'] < Q1_c2 - 1.5 * IQR_c2) | (data['c2'] > Q3_c2 + 1.5 * IQR_c2)]

# Αντικατάσταση outliers με τον μέσο όρο
data.loc[outliers_c1.index, 'c1'] = data['c1'].mean()
data.loc[outliers_c2.index, 'c2'] = data['c2'].mean()

print("Δεδομένα μετά την αντιμετώπιση των Outliers:")
print(data.head())

# Ταξινόμηση των δεδομένων βάσει χρονικής στιγμής (t)
data['t'] = pd.to_datetime(data['t'])
data = data.sort_values(by='t')
from scipy.interpolate import interp1d
# Define a function to interpolate a trajectory into a fixed number of points
def interpolate_trajectory(c1, c2, t, n_points=159):
    if len(c1) < 2:  # Not enough points to interpolate
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    
    # Normalize timestamps for interpolation
    norm_t = (t - t.min()) / (t.max() - t.min())
    
    # Interpolate latitude and longitude over a fixed number of evenly spaced points
    interpolated_times = np.linspace(0, 1, n_points)
    lat_interpolator = interp1d(norm_t, c1, kind='linear', fill_value="extrapolate")
    lon_interpolator = interp1d(norm_t, c2, kind='linear', fill_value="extrapolate")
    
    interpolated_c1 = lat_interpolator(interpolated_times)
    interpolated_c2 = lon_interpolator(interpolated_times)
    return interpolated_c1, interpolated_c2

# Επιλογή μιας συγκεκριμένης τροχιάς με βάση το ID
desired_traj_id = data['tid'].iloc[1]  # Επιλέγουμε το πρώτο ID για παράδειγμα
single_traj_data = data[data['tid'] == desired_traj_id]

# Extract the required columns
c1s=single_traj_data['c1'].values
c2s=single_traj_data['c2'].values
ts=single_traj_data['t'].values

# Interpolate the trajectory
n_points = 102
# Κλήση της συνάρτησης με τις κατάλληλες στήλες
interpolated_c1, interpolated_c2 = interpolate_trajectory(single_traj_data['c1'].values, single_traj_data['c2'].values, single_traj_data['t'].values)

# Plot the interpolated trajectory
plt.figure(figsize=(10, 6))
plt.plot(c1s, c2s, 'o-', label='Original Trajectory', alpha=0.7)
plt.plot(interpolated_c1, interpolated_c2, 'r-', label='Interpolated Trajectory', linewidth=2)
plt.xlabel('c1 (degrees)')
plt.ylabel('c2 (degrees)')
plt.title(f'Trajectory Interpolation for traj_id {desired_traj_id}')
plt.legend()
plt.grid()
plt.show()

# Apply to each trajectory
n_interpolation_points = 102  # Choose a fixed number of points for representation
trajectory_features = []

for tid, single_traj_data in data.groupby('tid'):
    c1s = single_traj_data['c1'].values
    c2s = single_traj_data['c2'].values
    ts = single_traj_data['t'].astype('int64').values  # Convert timestamps to int
    
    # Interpolate the trajectory
    interp_c1, interp_c2 = interpolate_trajectory(c1s,c2s,ts, n_points=n_interpolation_points)
    
    # Combine latitude and longitude features
    trajectory_vector = np.concatenate([interp_c1, interp_c2])
    trajectory_features.append({
        'tid': tid,
        'trajectory_vector': trajectory_vector
    })

# Convert to DataFrame
trajectory_features_df = pd.DataFrame(trajectory_features)
trajectory_features_df.info()
X = np.vstack(trajectory_features_df['trajectory_vector'].values)


# Standardize the features (important for PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Check the explained variance ratio for each number of PCA components
n_components = 102  # Test the first 10 components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Print the explained variance and cumulative variance
print("PCA Component-wise Explained Variance and Cumulative Variance:")
for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"Component {i+1}: Explained Variance = {var_ratio:.4f}, Cumulative Variance = {cum_var:.4f}")

# Step 2: Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.show()

import os
os.environ["OMP_NUM_THREADS"] = "1"
# Step 3: Find the optimal number of components using clustering (e.g., KMeans with Silhouette Score)
silhouette_scores = []

print("\nSilhouette Scores for KMeans Clustering:")
for n in range(1, n_components + 1):
    pca = PCA(n_components=n)
    X_reduced = pca.fit_transform(X_scaled)

    # Apply KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)

    # Calculate silhouette score
    sil_score = silhouette_score(X_reduced, clusters)
    silhouette_scores.append(sil_score)
    print(f"PCA Components: {n}, Silhouette Score: {sil_score:.4f}")

# Step 4: Plot silhouette scores to determine optimal components
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), silhouette_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Number of PCA Components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of PCA Components')
plt.grid()
plt.show()

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Προσθήκη των πρώτων 2 συνιστωσών στο DataFrame (δυναμικά)
for i in range(2):
    trajectory_features_df[f'pca_{i + 1}'] = X_reduced[:, i]

# Εμφάνιση του DataFrame με τις πρώτες δύο συνιστώσες
print(trajectory_features_df[['tid', 'pca_1', 'pca_2']].head())

# Κανονικοποίηση του X_reduced 
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_reduced)  

# Εκτύπωση πληροφοριών για έλεγχο
print("Shape of reduced data:", X_reduced.shape)
print("Shape of normalized data:", X_normalized.shape)

print(trajectory_features_df)


# Range of cluster numbers to try
cluster_range = range(1, min(10, len(X_normalized) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=30)
    kmeans.fit(X_normalized)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Run K-Means with a predefined number of clusters
n_clusters =4 # Adjust this based on your dataset and goal
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
clusters = kmeans.fit_predict(X_normalized)

# Add the cluster labels back to the trajectory features DataFrame
trajectory_features_df['cluster'] = clusters

# Inspect the clustering results
print(trajectory_features_df[['tid', 'cluster']].head())

# Calculate silhouette score
sil_score = silhouette_score(X_normalized, clusters)
print(f"Silhouette Score: {sil_score}")

# Visualize the clusters in 2D using PCA components
plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    cluster_points = trajectory_features_df[trajectory_features_df['cluster'] == cluster_id]
    plt.scatter(cluster_points['pca_1'], cluster_points['pca_2'], label=f"Cluster {cluster_id}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("K-Means Clusters of Trajectories")
plt.show()

# Create a list to store results
results = []

# Range of k values to test
k_range = range(2, 11)

# Size of subplots for visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Convert to 1D list for easy indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = kmeans.fit_predict(X_normalized)  # Use the normalized trajectory vectors
    
    # Calculate Silhouette Coefficient (only if there are at least 2 clusters)
    silhouette = silhouette_score(X_normalized, labels)
    
    # Calculate Sum of Squared Errors (SSE or Inertia)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualize the clusters in PCA space (use PCA components for visualization)
    ax = axes[idx]
    ax.scatter(trajectory_features_df['pca_1'], 
               trajectory_features_df['pca_2'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids (transformed to PCA space)
    centroids = kmeans.cluster_centers_
    centroids_pca = PCA(n_components=2).fit_transform(centroids)  # Transform to PCA space
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    ax.grid(True)

# Hide empty plots if there are fewer than 9 clusters
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional visualization of metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error (SSE)
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()

#DBSCAN
# Step 1: Scale the data and reduce dimensionality using PCA (already done above)
# Ensure X_pca_reduced is the PCA-reduced data (e.g., 2 components)
# Step 2: Run DBSCAN with varying eps and min_samples
#eps_values = np.arange(0.1, 5, 0.2)  # Adjust the range based on dataset scale
#eps_values = np.arange(0.05, 2, 0.05)
#min_samples_values = [2, 5, 10, 15, 17, 20,25]  

eps_values = np.arange(0.6, 0.85, 0.05)
min_samples_values = [10, 12, 13, 15, 17]

best_eps = None
best_silhouette = -1
best_dbscan = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_normalized)

        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_normalized, labels)
            print(f"DBSCAN: eps={round(eps, 2)}, min_samples={min_samples}, silhouette={silhouette:.4f}")

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_dbscan = dbscan
                best_labels = labels

if best_labels is None:
    print("No valid clustering found. Try adjusting eps or scaling data.")
    exit()

print(f"Best DBSCAN: eps={round(best_eps, 2)}, silhouette={best_silhouette:.4f}")

# Step 3: Visualize the clusters from the best DBSCAN model
plt.figure(figsize=(10, 6))
unique_labels = np.unique(best_labels)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    label_points = X_normalized[best_labels == label]
    if label == -1:  # Noise points
        plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
    else:
        plt.scatter(label_points[:, 0], label_points[:, 1], c=[color], label=f'Cluster {label}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f"DBSCAN Clustering (eps={round(best_eps, 2)})")
plt.legend()
plt.show()
noise_ratio = np.sum(labels == -1) / len(labels)
print(f"Noise ratio: {noise_ratio:.2%}")


#try2
# Initialize OPTICS
min_samples_values = [5, 10,15,20,25, 30]  # Varying min_samples values
best_silhouette = -1
best_min_samples = None
best_labels = None

# Loop over min_samples to test different configurations
for min_samples in min_samples_values:
    optics = OPTICS(min_samples=min_samples, metric='minkowski',xi=0.03, min_cluster_size=0.3)
    labels = optics.fit_predict(X_normalized)  # X_normalized: your scaled trajectory vectors
    
    # Skip silhouette score calculation if all points are noise or in one cluster
    if len(set(labels)) > 1:  # Ensure more than one cluster exists
        silhouette = silhouette_score(X_normalized, labels)
        print(f"OPTICS: min_samples={min_samples}, silhouette={silhouette}")
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_min_samples = min_samples
            best_labels = labels

if best_labels is not None:
    print(f"Best OPTICS: min_samples={best_min_samples}, silhouette={best_silhouette}")

    # Visualization of OPTICS results
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        if label == -1:  # Noise points
            plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
        else:
            plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Cluster {label}')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"OPTICS Clustering (min_samples={best_min_samples})")
    plt.legend()
    plt.show()
else:
    print("OPTICS could not find any meaningful clusters with the given parameters.")

# Optional: Extract Reachability Plot
reachability = optics.reachability_[optics.ordering_]
plt.figure(figsize=(10, 6))
plt.plot(reachability)
plt.title("OPTICS Reachability Plot")
plt.xlabel("Sample Index")
plt.ylabel("Reachability Distance")
plt.show()

#ΤΡΥ3
# Updated parameter ranges
min_samples_values = [5, 10,12, 15, 20, 25]  
xi_values = [0.01,0.03, 0.05,0.07, 0.1, 0.2]  
min_cluster_size_values = [0.02, 0.05,0.08, 0.1,0.15, 0.2]  

best_silhouette = -1
best_params = None
best_labels = None

# Test multiple configurations
for min_samples in min_samples_values:
    for xi in xi_values:
        for min_cluster_size in min_cluster_size_values:
            optics = OPTICS(
                min_samples=min_samples,
                metric='euclidean',
                xi=xi,
                min_cluster_size=min_cluster_size,
            )
            labels = optics.fit_predict(X_normalized)
            
            # Skip silhouette score calculation if all points are noise or in one cluster
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_normalized, labels)
                print(f"OPTICS: min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}, silhouette={silhouette}")
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_params = (min_samples, xi, min_cluster_size)
                    best_labels = labels

if best_labels is not None:
    min_samples, xi, min_cluster_size = best_params
    print(f"Best OPTICS: min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}, silhouette={best_silhouette}")
    
    # Visualization of OPTICS results
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        if label == -1:  # Noise points
            plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
        else:
            plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Cluster {label}')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"OPTICS Clustering (min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size})")
    plt.legend()
    plt.show()
else:
    print("OPTICS could not find any meaningful clusters with the given parameters.")

# Reachability plot
reachability = optics.reachability_[optics.ordering_]
plt.figure(figsize=(10, 6))
plt.plot(reachability)
plt.title("OPTICS Reachability Plot")
plt.xlabel("Sample Index")
plt.ylabel("Reachability Distance")
plt.show()


# === Σύγκριση DBSCAN και OPTICS σε 2x2 layout ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- DBSCAN Scatter (πάνω αριστερά) ---
unique_labels_dbscan = np.unique(best_labels)  # προσοχή: αυτό τώρα είναι για DBSCAN, κράτα το ξεχωριστά!
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels_dbscan)))
for label, color in zip(unique_labels_dbscan, colors):
    points = X_normalized[best_labels == label]
    if label == -1:
        axes[0, 0].scatter(points[:, 0], points[:, 1], c="gray", alpha=0.5, label="Noise")
    else:
        axes[0, 0].scatter(points[:, 0], points[:, 1], c=[color], label=f"Cluster {label}")
axes[0, 0].set_title(f"DBSCAN (eps={round(best_eps,2)})")
axes[0, 0].set_xlabel("PCA Component 1")
axes[0, 0].set_ylabel("PCA Component 2")
axes[0, 0].legend()

# --- OPTICS Scatter (πάνω δεξιά) ---
unique_labels_optics = np.unique(best_labels)
for label in unique_labels_optics:
    points = X_normalized[best_labels == label]
    if label == -1:
        axes[0, 1].scatter(points[:, 0], points[:, 1], c="gray", alpha=0.5, label="Noise")
    else:
        axes[0, 1].scatter(points[:, 0], points[:, 1], label=f"Cluster {label}")
min_samples, xi, min_cluster_size = best_params
axes[0, 1].set_title(f"OPTICS (min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size})")
axes[0, 1].set_xlabel("PCA Component 1")
axes[0, 1].set_ylabel("PCA Component 2")
axes[0, 1].legend()

# --- OPTICS Reachability Plot (κάτω αριστερά) ---
reachability = optics.reachability_[optics.ordering_]
axes[1, 0].plot(reachability)
axes[1, 0].set_title("OPTICS Reachability Plot")
axes[1, 0].set_xlabel("Sample Index")
axes[1, 0].set_ylabel("Reachability Distance")

# --- Κενό subplot (κάτω δεξιά) ---
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

----------------------------------------------------------------------------------------------------------

#Dataset London Landings
data = pd.read_csv('londonlandingsync_all.csv')  
data.columns = ["obj_id","traj_id","timestamp","lon(deg)","lat(deg)","alt(meters)"]

data.head()

#Missing Values
data['lon(deg)'] = data['lon(deg)'].fillna(data['lon(deg)'].mean())
data['lat(deg)'] = data['lat(deg)'].fillna(data['lat(deg)'].mean())
data['alt(meters)'] = data['alt(meters)'].fillna(data['alt(meters)'].mean())

print(data.head())

# Outliers
Q1_lon = data['lon(deg)'].quantile(0.25)
Q3_lon = data['lon(deg)'].quantile(0.75)
IQR_lon = Q3_lon - Q1_lon
Q1_lat = data['lat(deg)'].quantile(0.25)
Q3_lat = data['lat(deg)'].quantile(0.75)
IQR_lat = Q3_lat - Q1_lat

outliers_lon = data[(data['lon(deg)'] < Q1_lon - 1.5 * IQR_lon) | (data['lon(deg)'] > Q3_lon + 1.5 * IQR_lon)]
outliers_lat = data[(data['lat(deg)'] < Q1_lat - 1.5 * IQR_lat) | (data['lat(deg)'] > Q3_lat + 1.5 * IQR_lat)]

data.loc[outliers_lon.index, 'lon(deg)'] = data['lon(deg)'].mean()
data.loc[outliers_lat.index, 'lat(deg)'] = data['lat(deg)'].mean()

# Sort data by timestamp if not already sorted
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')


# Apply a moving average with a window size (e.g., 5 points) to smooth longitude and latitude
window_size = 3
data['smoothed_longitude'] = data['lon(deg)'].rolling(window=window_size, center=True).mean()
data['smoothed_latitude'] = data['lat(deg)'].rolling(window=window_size, center=True).mean()

# Drop any rows with NaN values created by the moving average
smoothed_data = data.dropna(subset=['smoothed_longitude', 'smoothed_latitude'])
# Display the smoothed trajectory
print(smoothed_data[['smoothed_longitude', 'smoothed_latitude']].head())

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(data['lon(deg)'], data['lat(deg)'], s=0.5, c='red', label='Original')
ax[0].set_title("Πριν την Εξομάλυνση")
ax[0].set_xlabel("Longitude (deg)")
ax[0].set_ylabel("Latitude (deg)")
ax[0].legend()

ax[1].scatter(smoothed_data['smoothed_longitude'],
              smoothed_data['smoothed_latitude'],
              s=0.5, c='blue', label='Smoothed')
ax[1].set_title("Μετά την Εξομάλυνση")
ax[1].set_xlabel("Longitude (deg)")
ax[1].set_ylabel("Latitude (deg)")
ax[1].legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['lon(deg)'], data['lat(deg)'], s=10, alpha=0.6, c='steelblue')
plt.title('Θέσεις London Landing')
plt.xlabel('lon(deg)')
plt.ylabel('lat(deg)')
plt.grid(True)
plt.show()

#Vectorisation
# Function to calculate bearing (direction) between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# calculate for each flight
flight_vectors = []

# Group data by flight
for obj_id, traj_data in smoothed_data.groupby('obj_id'):
    speeds = []
    bearings = []
    
    for i in range(1, len(traj_data)):
        point1 = (traj_data.iloc[i-1]['smoothed_latitude'], traj_data.iloc[i-1]['smoothed_longitude'])
        point2 = (traj_data.iloc[i]['smoothed_latitude'], traj_data.iloc[i]['smoothed_longitude'])
        
        # calculate distance and time
        distance = geodesic(point1, point2).meters
        time_delta = (traj_data.iloc[i]['timestamp'] - traj_data.iloc[i-1]['timestamp']).total_seconds()
        
        # calculate speed and bearing  
        speed = distance / time_delta if time_delta > 0 else 0
        speeds.append(speed)
        
        bearing = calculate_bearing(point1[0], point1[1], point2[0], point2[1])
        bearings.append(bearing)
    
    # Calculation of average speed and bearing for the flight
    avg_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)
    speed_25th = np.percentile(speeds, 25)
    speed_50th = np.percentile(speeds, 50)
    speed_75th = np.percentile(speeds, 75)
    speed_range = max_speed - min_speed
   
    avg_bearing = np.mean(bearings)
    std_bearing = np.std(bearings)
    min_bearing = np.min(bearings)
    max_bearing = np.max(bearings)
    bearing_range = max_bearing - min_bearing
    
    # Cap the average speed at 6500
    if speed_range >50000:
        speed_range = 50000
   
    # Adding the flight vector to the list
    flight_vectors.append({
        'obj_id': obj_id,
        'avg_speed': avg_speed,'std_speed': std_speed,'min_speed': min_speed , 'max_speed': max_speed ,
        'speed_25th': speed_25th , 'speed_50th': speed_50th , 'speed_75th': speed_75th , 'speed_range': speed_range ,
        'avg_bearing': avg_bearing,'std_bearing': std_bearing, 'min_bearing': min_bearing, 'max_bearing' : max_bearing ,
        'bearing_range' : bearing_range
    })

# Convert to DataFrame for easier viewing
flight_vectors_df = pd.DataFrame(flight_vectors)
print(len(flight_vectors))

# Extract features for k-means clustering
flight_features = flight_vectors_df[['obj_id','min_speed', 'max_speed','speed_50th','speed_range','min_bearing','max_bearing','bearing_range','avg_bearing', 'std_bearing']]


# Range of cluster numbers to try
cluster_range = range(1, min(10, len(flight_features) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(flight_features)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Check for any NaN or infinite values in segment_features
print("Missing values per column:\n", flight_features.isna().sum())
print("Infinite values present:", np.isinf(flight_features).values.any())
# If any NaN or infinite values are found, drop or replace them
flight_features =flight_features.dropna()  # Drop rows with NaN (if any)

# KMeans 
# k=2
optimal_k = 2 
clustering_features = ['min_speed', 'max_speed', 'speed_50th', 'speed_range', 
                       'min_bearing', 'max_bearing', 'bearing_range','avg_bearing', 'std_bearing']
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
flight_vectors_df['cluster'] = kmeans.fit_predict(flight_vectors_df[clustering_features])

# Display the clustered segments
print(flight_vectors_df)

centroids = kmeans.cluster_centers_

labels = kmeans.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of iterations: %d' % kmeans.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(flight_features, labels))
print("Mean Squared Error: %0.3f" % kmeans.inertia_)  # returns the SSE value


# Planning flights based on their cluster
plt.figure(figsize=(10, 5))
plt.scatter(flight_vectors_df['speed_range'], 
            flight_vectors_df['avg_bearing'], 
            c=flight_vectors_df['cluster'], 
            cmap='viridis', 
            marker='o',
            s=100,  
            edgecolor='k')

# Adding centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,clustering_features.index('speed_range')], centroids[:,clustering_features.index('avg_bearing')], 
            c='red', 
            marker='X', 
            s=200,  
            label='Centroids')

plt.xlabel('Range Speed')
plt.ylabel('Average Bearing')
plt.title('Cluster Separation of Flight Paths')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# KMeans 
# k=4
optimal_k = 4
clustering_features = ['min_speed', 'max_speed', 'speed_50th', 'speed_range', 
                       'min_bearing', 'max_bearing', 'bearing_range','avg_bearing', 'std_bearing']
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
flight_vectors_df['cluster'] = kmeans.fit_predict(flight_vectors_df[clustering_features])

# Display the clustered segments
print(flight_vectors_df)

centroids = kmeans.cluster_centers_

labels = kmeans.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of iterations: %d' % kmeans.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(flight_features, labels))
print("Mean Squared Error: %0.3f" % kmeans.inertia_)  # returns the SSE value


# Planning flights based on their cluster
plt.figure(figsize=(10, 5))
plt.scatter(flight_vectors_df['speed_range'], 
            flight_vectors_df['avg_bearing'], 
            c=flight_vectors_df['cluster'], 
            cmap='viridis', 
            marker='o',
            s=100,  
            edgecolor='k')

# Adding centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,clustering_features.index('speed_range')], centroids[:,clustering_features.index('avg_bearing')], 
            c='red', 
            marker='X', 
            s=200,  
            label='Centroids')

plt.xlabel('Range Speed')
plt.ylabel('Average Bearing')
plt.title('Cluster Separation of Flight Paths')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# Define clustering features
clustering_features = ['min_speed', 'max_speed', 'speed_50th', 'speed_range', 
                       'min_bearing', 'max_bearing', 'bearing_range','avg_bearing', 'std_bearing']

plt.figure(figsize=(10, 6))
plt.scatter(
    flight_vectors_df['speed_range'], 
    flight_vectors_df['avg_bearing'],
    c=flight_vectors_df['cluster'],      # Χρώμα ανά cluster
    cmap='tab10',
    s=30,
    alpha=0.7,
    edgecolor='k'
)
plt.xlabel('speed_range')
plt.ylabel('avg_bearing')
plt.title('Clusters στον Γεωγραφικό Χώρο')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Create list to store results
results = []

# Range of k values
k_range = range(2, 11)

# Subplot grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Flatten into 1D array for easier indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flight_vectors_df[clustering_features])
    
    # Calculate Silhouette Coefficient
    silhouette = silhouette_score(flight_vectors_df[clustering_features], labels)
    
    # Calculate Mean Squared Error (SSE)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualization of clusters
    ax = axes[idx]
    ax.scatter(flight_vectors_df['speed_range'], 
               flight_vectors_df['avg_bearing'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, clustering_features.index('speed_range')], 
               centroids[:, clustering_features.index('avg_bearing')], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('Range Speed')
    ax.set_ylabel('Average Bearing')
    ax.legend()
    ax.grid(True)

# Hide unused subplot axes
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional Visualization of Metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()


# OPTICS Clustering
clust = OPTICS(min_samples=15, metric='euclidean', xi=0.01, min_cluster_size=0.2)

# Apply OPTICS on the selected features
clust.fit(flight_vectors_df[['min_speed','avg_bearing', 'std_bearing']])

# Generate DBSCAN labels at different epsilon values
labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=1)
labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=5)

# Prepare data for the reachability plot
space = np.arange(len(flight_vectors_df))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

# Clean reachability array to remove NaN or Inf values
valid_reachability = reachability[np.isfinite(reachability)]

# Plotting
plt.figure(figsize=(15, 10))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Set axis limits dynamically (avoid NaN or Inf errors)
if len(valid_reachability) > 0:  # Ensure there are valid values
    ax1.set_ylim(0, valid_reachability.max() + 10)
else:
    ax1.set_ylim(0, 10)  # Fallback y-axis limit

ax1.set_xlim(0, len(space))

# Set static limits for clustering plots
ax2.set_xlim(0, 1000)
ax2.set_ylim(100, 300)
ax3.set_xlim(0, 1000)
ax3.set_ylim(100, 300)
ax4.set_xlim(0, 1000)
ax4.set_ylim(100, 300)

# Reachability Plot
colors = ['g.', 'r.', 'y.', 'b.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)

# Plot noise points
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS Automatic Clustering
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[clust.labels_ == klass]
    ax2.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=1)
ax2.plot(flight_vectors_df[clust.labels_ == -1]['min_speed'], 
         flight_vectors_df[clust.labels_ == -1]['avg_bearing'], 'k.', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN Clustering with eps=23
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[labels_100 == klass]
    ax3.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=0.3)
ax3.plot(flight_vectors_df[labels_100 == -1]['min_speed'], 
         flight_vectors_df[labels_100 == -1]['avg_bearing'], 'k+', alpha=0.1)
ax3.set_title('Clustering at 1 epsilon cut\nDBSCAN')

# DBSCAN Clustering with eps=26
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[labels_120 == klass]
    ax4.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=0.3)
ax4.plot(flight_vectors_df[labels_120 == -1]['min_speed'], 
         flight_vectors_df[labels_120 == -1]['avg_bearing'], 'k+', alpha=0.1)
ax4.set_title('Clustering at 5 epsilon cut\nDBSCAN')

# Finalize Layout
plt.tight_layout()
plt.show()

# Λίστες για αποθήκευση αποτελεσμάτων
results_optics = []

# Υπολογισμός Silhouette και SSE για OPTICS
for epsilon, labels in zip([1, 5], [labels_100, labels_120]):
    # Εξαιρούμε σημεία με θόρυβο (label = -1)
    valid_points = flight_vectors_df[labels != -1]
    valid_labels = labels[labels != -1]
    
    if len(np.unique(valid_labels)) > 1:  # Υπολογίζουμε Silhouette μόνο αν έχουμε >1 cluster
        silhouette = silhouette_score(valid_points[[ 'min_speed','avg_bearing', 'std_bearing']], valid_labels)
    else:
        silhouette = np.nan  # Δεν ορίζεται για ένα μόνο cluster
    
    # Υπολογισμός SSE (άθροισμα αποστάσεων από τα "κεντροειδή" κάθε cluster)
    sse = 0
    for cluster in np.unique(valid_labels):
        cluster_points = valid_points[valid_labels == cluster]
        centroid = cluster_points[[ 'min_speed','avg_bearing', 'std_bearing']].mean(axis=0)
        sse += np.sum((cluster_points[[ 'min_speed','avg_bearing', 'std_bearing']] - centroid) ** 2)
    
    # Αποθήκευση αποτελεσμάτων
    results_optics.append({
        'Epsilon': epsilon,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': sse
    })

# Μετατροπή σε DataFrame για καλύτερη εμφάνιση
results_optics_df = pd.DataFrame(results_optics)

# Ρύθμιση για εμφάνιση όλων των στηλών χωρίς περικοπή
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Εμφάνιση αποτελεσμάτων
print(results_optics_df)

# Define different parameter scenarios for OPTICS clustering and DBSCAN eps values

scenarios = [
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.2, 'eps_1': 3, 'eps_2': 7.5},
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.2, 'eps_1': 3, 'eps_2': 8},
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.25, 'eps_1': 3, 'eps_2': 7},
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.15, 'eps_1': 3, 'eps_2': 7.5},
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.15, 'eps_1': 2.5, 'eps_2': 8.5},
    {'min_samples': 15, 'xi': 0.02, 'min_cluster_size': 0.3, 'eps_1': 2.5, 'eps_2': 7},
    {'min_samples': 12, 'xi': 0.01, 'min_cluster_size': 0.18, 'eps_1': 3, 'eps_2': 7.2},
    {'min_samples': 20, 'xi': 0.01, 'min_cluster_size': 0.2, 'eps_1': 3, 'eps_2': 7},
    {'min_samples': 20, 'xi': 0.02, 'min_cluster_size': 0.2, 'eps_1': 3, 'eps_2': 7.5},
    {'min_samples': 20, 'xi': 0.02, 'min_cluster_size': 0.25, 'eps_1': 3, 'eps_2': 7},
    ]
def run_optics_scenarios(scenarios,data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['min_speed','avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        silhouette_avg_1 = silhouette_score(data[['min_speed','avg_bearing', 'std_bearing']], labels_100)
        silhouette_avg_2 = silhouette_score(data[['min_speed','avg_bearing', 'std_bearing']], labels_120)
        # Calculate silhouette scores only if there are at least 2 clusters
        
        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results

def run_optics_scenarios(scenarios, data):
    results = []
    for i, params in enumerate(scenarios):
        clust = OPTICS(min_samples=params['min_samples'], metric='euclidean', xi=params['xi'], min_cluster_size=params['min_cluster_size'])
        clust.fit(data[['min_speed', 'avg_bearing', 'std_bearing']])
        
        labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_1'])
        labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=params['eps_2'])
        
        # Ensure silhouette scores are calculated only for valid clusterings
        if len(set(labels_100)) > 1:
            silhouette_avg_1 = silhouette_score(data[['min_speed', 'avg_bearing', 'std_bearing']], labels_100)
        else:
            silhouette_avg_1 = -1  # Invalid clustering (only one cluster)
        
        if len(set(labels_120)) > 1:
            silhouette_avg_2 = silhouette_score(data[['min_speed', 'avg_bearing', 'std_bearing']], labels_120)
        else:
            silhouette_avg_2 = -1  # Invalid clustering (only one cluster)

        results.append({
            'scenario': i + 1,
            'params': params,
            'silhouette_avg_1': silhouette_avg_1,
            'silhouette_avg_2': silhouette_avg_2
        })
    
    return results
results = run_optics_scenarios(scenarios, flight_vectors_df)
for res in results:
    print(res)
    
    results = run_optics_scenarios(scenarios, flight_vectors_df)

# Find the best result based on silhouette_avg_1 and silhouette_avg_2
best_result_1 = max(results, key=lambda x: x['silhouette_avg_1'])
best_result_2 = max(results, key=lambda x: x['silhouette_avg_2'])

# Print the best results
print("Best Result for eps_1:")
print(f"Scenario: {best_result_1['scenario']}")
print(f"Parameters: {best_result_1['params']}")
print(f"Silhouette Score: {best_result_1['silhouette_avg_1']}")

print("\nBest Result for eps_2:")
print(f"Scenario: {best_result_2['scenario']}")
print(f"Parameters: {best_result_2['params']}")
print(f"Silhouette Score: {best_result_2['silhouette_avg_2']}")


# OPTICS Clustering
clust = OPTICS(min_samples=10, metric='euclidean', xi=0.01, min_cluster_size=0.15)

# Apply OPTICS on the selected features
clust.fit(flight_vectors_df[['min_speed','avg_bearing', 'std_bearing']])

# Generate DBSCAN labels at different epsilon values
labels_100 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2.5)
labels_120 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=8.5)

# Prepare data for the reachability plot
space = np.arange(len(flight_vectors_df))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

# Clean reachability array to remove NaN or Inf values
valid_reachability = reachability[np.isfinite(reachability)]

# Plotting
plt.figure(figsize=(15, 10))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Set axis limits dynamically (avoid NaN or Inf errors)
if len(valid_reachability) > 0:  # Ensure there are valid values
    ax1.set_ylim(0, valid_reachability.max() + 10)
else:
    ax1.set_ylim(0, 10)  # Fallback y-axis limit

ax1.set_xlim(0, len(space))

# Set static limits for clustering plots
ax2.set_xlim(0, 1000)
ax2.set_ylim(100, 300)
ax3.set_xlim(0, 1000)
ax3.set_ylim(100, 300)
ax4.set_xlim(0, 1000)
ax4.set_ylim(100, 300)

# Reachability Plot
colors = ['g.', 'r.', 'y.', 'b.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)

# Plot noise points
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS Automatic Clustering
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[clust.labels_ == klass]
    ax2.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=1)
ax2.plot(flight_vectors_df[clust.labels_ == -1]['min_speed'], 
         flight_vectors_df[clust.labels_ == -1]['avg_bearing'], 'k.', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN Clustering with eps=23
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[labels_100 == klass]
    ax3.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=0.3)
ax3.plot(flight_vectors_df[labels_100 == -1]['min_speed'], 
         flight_vectors_df[labels_100 == -1]['avg_bearing'], 'k+', alpha=0.1)
ax3.set_title('Clustering at 2.5 epsilon cut\nDBSCAN')

# DBSCAN Clustering with eps=26
for klass, color in zip(range(0, 5), colors):
    Xk = flight_vectors_df[labels_120 == klass]
    ax4.plot(Xk['min_speed'], Xk['avg_bearing'], color, alpha=0.3)
ax4.plot(flight_vectors_df[labels_120 == -1]['min_speed'], 
         flight_vectors_df[labels_120 == -1]['avg_bearing'], 'k+', alpha=0.1)
ax4.set_title('Clustering at 8.5 epsilon cut\nDBSCAN')

# Finalize Layout
plt.tight_layout()
plt.show()
-------------------------------------------------------------------------------------------------------------
#methodos PCA 

#Dataset
data = pd.read_csv('londonlandingsync_all.csv')  
data.columns = ["obj_id","traj_id","timestamp","lon(deg)","lat(deg)","alt(meters)"]

data.head()

#Missing Values
data['lon(deg)'] = data['lon(deg)'].fillna(data['lon(deg)'].mean())
data['lat(deg)'] = data['lat(deg)'].fillna(data['lat(deg)'].mean())
data['alt(meters)'] = data['alt(meters)'].fillna(data['alt(meters)'].mean())

print(data.head())


# Outliers
Q1_lon = data['lon(deg)'].quantile(0.25)
Q3_lon = data['lon(deg)'].quantile(0.75)
IQR_lon = Q3_lon - Q1_lon
Q1_lat = data['lat(deg)'].quantile(0.25)
Q3_lat = data['lat(deg)'].quantile(0.75)
IQR_lat = Q3_lat - Q1_lat

outliers_lon = data[(data['lon(deg)'] < Q1_lon - 1.5 * IQR_lon) | (data['lon(deg)'] > Q3_lon + 1.5 * IQR_lon)]
outliers_lat = data[(data['lat(deg)'] < Q1_lat - 1.5 * IQR_lat) | (data['lat(deg)'] > Q3_lat + 1.5 * IQR_lat)]

data.loc[outliers_lon.index, 'lon(deg)'] = data['lon(deg)'].mean()
data.loc[outliers_lat.index, 'lat(deg)'] = data['lat(deg)'].mean()

# Sort data by timestamp if not already sorted
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')

# Apply a moving average with a window size (e.g., 5 points) to smooth longitude and latitude
window_size = 5
data['smoothed_longitude'] = data['lon(deg)'].rolling(window=window_size, center=True).mean()
data['smoothed_latitude'] = data['lat(deg)'].rolling(window=window_size, center=True).mean()

# Drop any rows with NaN values created by the moving average
smoothed_data = data.dropna(subset=['smoothed_longitude', 'smoothed_latitude'])
# Display the smoothed trajectory
print(smoothed_data[['smoothed_longitude', 'smoothed_latitude']].head())


# Define a function to interpolate a trajectory into a fixed number of points
def interpolate_trajectory(latitudes, longitudes, timestamps, n_points=5):
    if len(latitudes) < 2:  # Not enough points to interpolate
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    
    # Normalize timestamps for interpolation
    norm_timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    # Interpolate latitude and longitude over a fixed number of evenly spaced points
    interpolated_times = np.linspace(0, 1, n_points)
    lat_interpolator = interp1d(norm_timestamps, latitudes, kind='linear', fill_value="extrapolate")
    lon_interpolator = interp1d(norm_timestamps, longitudes, kind='linear', fill_value="extrapolate")
    
    interpolated_lats = lat_interpolator(interpolated_times)
    interpolated_lons = lon_interpolator(interpolated_times)
    return interpolated_lats, interpolated_lons

# Select a specific trajectory ID 
desired_traj_id = data['obj_id'].iloc[100] 
single_traj_data = data[data['obj_id'] == desired_traj_id]

# Extract the required columns
latitudes = single_traj_data['lat(deg)'].values
longitudes = single_traj_data['lon(deg)'].values
timestamps = pd.to_datetime(single_traj_data['timestamp']).astype('int64') // 10**9  # Convert timestamps to UNIX time


# Interpolate the trajectory
n_points = 5
interpolated_lats, interpolated_lons = interpolate_trajectory(latitudes, longitudes, timestamps, n_points=n_points)

# Plot the interpolated trajectory
plt.figure(figsize=(10, 6))
plt.plot(longitudes, latitudes, 'o-', label='Original Trajectory', alpha=0.7)
plt.plot(interpolated_lons, interpolated_lats, 'r-', label='Interpolated Trajectory', linewidth=2)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title(f'Trajectory Interpolation for traj_id {desired_traj_id}')
plt.legend()
plt.grid()
plt.show()

# Apply to each trajectory
n_interpolation_points = 5  # Choose a fixed number of points for representation
trajectory_features = []

for obj_id, traj_data in smoothed_data.groupby('obj_id'):
    latitudes = traj_data['smoothed_latitude'].values
    longitudes = traj_data['smoothed_longitude'].values
    timestamps = traj_data['timestamp'].astype('int64').values  # Convert timestamps to int
    
    # Interpolate the trajectory
    interp_lats, interp_lons = interpolate_trajectory(latitudes, longitudes, timestamps, n_points=n_interpolation_points)
    
    # Combine latitude and longitude features
    trajectory_vector = np.concatenate([interp_lats, interp_lons])
    trajectory_features.append({
        'obj_id': obj_id,
        'trajectory_vector': trajectory_vector
    })

# Convert to DataFrame
trajectory_features_df = pd.DataFrame(trajectory_features)
trajectory_features_df.info()
X = np.vstack(trajectory_features_df['trajectory_vector'].values)

# Standardize the features (important for PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Check the explained variance ratio for each number of PCA components
n_components = 10  # Test the first 10 components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Print the explained variance and cumulative variance
print("PCA Component-wise Explained Variance and Cumulative Variance:")
for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"Component {i+1}: Explained Variance = {var_ratio:.4f}, Cumulative Variance = {cum_var:.4f}")

# Step 2: Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.show()

# Step 3: Find the optimal number of components using clustering (e.g., KMeans with Silhouette Score)
silhouette_scores = []

print("\nSilhouette Scores for KMeans Clustering:")
for n in range(1, n_components + 1):
    pca = PCA(n_components=n)
    X_reduced = pca.fit_transform(X_scaled)

    # Apply KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)

    # Calculate silhouette score
    sil_score = silhouette_score(X_reduced, clusters)
    silhouette_scores.append(sil_score)
    print(f"PCA Components: {n}, Silhouette Score: {sil_score:.4f}")

# Step 4: Plot silhouette scores to determine optimal components
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_components+1), silhouette_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Number of PCA Components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of PCA Components')
plt.grid()
plt.show()


pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X)

# Προσθήκη των πρώτων 4 συνιστωσών στο DataFrame (δυναμικά)
for i in range(4):
    trajectory_features_df[f'pca_{i + 1}'] = X_reduced[:, i]

# Εμφάνιση του DataFrame με τις πρώτες δύο συνιστώσες
print(trajectory_features_df[['obj_id', 'pca_1', 'pca_2']].head())

# Κανονικοποίηση του X_reduced 
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_reduced)  

# Εκτύπωση πληροφοριών για έλεγχο
print("Shape of reduced data:", X_reduced.shape)
print("Shape of normalized data:", X_normalized.shape)

print(trajectory_features_df)


# Range of cluster numbers to try
cluster_range = range(1, min(10, len(X_normalized) + 1))
inertia_values = []

# Calculate inertia for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=30)
    kmeans.fit(X_normalized)
    inertia_values.append(kmeans.inertia_)

# Plot inertia to find elbow point
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Segment Clustering)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Run K-Means with a predefined number of clusters
n_clusters =4 # Adjust this based on your dataset and goal
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
clusters = kmeans.fit_predict(X_normalized)

# Add the cluster labels back to the trajectory features DataFrame
trajectory_features_df['cluster'] = clusters

# Inspect the clustering results
print(trajectory_features_df[['obj_id', 'cluster']].head())

# Calculate silhouette score
sil_score = silhouette_score(X_normalized, clusters)
print(f"Silhouette Score: {sil_score}")

# Visualize the clusters in 2D using PCA components
plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    cluster_points = trajectory_features_df[trajectory_features_df['cluster'] == cluster_id]
    plt.scatter(cluster_points['pca_1'], cluster_points['pca_2'], label=f"Cluster {cluster_id}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("K-Means Clusters of Trajectories")
plt.show()

# Create a list to store results
results = []

# Range of k values to test
k_range = range(2, 11)

# Size of subplots for visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Convert to 1D list for easy indexing

for idx, k in enumerate(k_range):
    # Run KMeans for each k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = kmeans.fit_predict(X_normalized)  # Use the normalized trajectory vectors
    
    # Calculate Silhouette Coefficient (only if there are at least 2 clusters)
    silhouette = silhouette_score(X_normalized, labels)
    
    # Calculate Sum of Squared Errors (SSE or Inertia)
    mse = kmeans.inertia_
    
    # Store the results
    results.append({
        'k': k,
        'Silhouette Coefficient': silhouette,
        'Mean Squared Error (SSE)': mse
    })
    
    # Visualize the clusters in PCA space (use PCA components for visualization)
    ax = axes[idx]
    ax.scatter(trajectory_features_df['pca_1'], 
               trajectory_features_df['pca_2'], 
               c=labels, 
               cmap='viridis', 
               marker='o', 
               s=50, 
               edgecolor='k', 
               label=f'k={k}')
    
    # Add centroids (transformed to PCA space)
    centroids = kmeans.cluster_centers_
    centroids_pca = PCA(n_components=2).fit_transform(centroids)  # Transform to PCA space
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', 
               marker='X', 
               s=200, 
               label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    ax.grid(True)

# Hide empty plots if there are fewer than 9 clusters
for idx in range(len(k_range), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Optional visualization of metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Mean Squared Error (SSE)
ax1.plot(results_df['k'], results_df['Mean Squared Error (SSE)'], marker='o', label='SSE', color='blue')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Mean Squared Error (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Silhouette Coefficient
ax2 = ax1.twinx()
ax2.plot(results_df['k'], results_df['Silhouette Coefficient'], marker='s', label='Silhouette Coefficient', color='green')
ax2.set_ylabel('Silhouette Coefficient', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('KMeans Performance Metrics (k = 2 to 10)')
plt.show()

#DBSCAN
eps_values = np.arange(5, 15.5, 0.5)
min_samples_values = [2,5, 8, 10, 12] # Different density thresholds
best_eps = None
best_silhouette = -1
best_dbscan = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_normalized)

        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_normalized, labels)
            print(f"DBSCAN: eps={eps}, min_samples={min_samples}, silhouette={silhouette:.4f}")
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_dbscan = dbscan
                best_labels = labels

if best_labels is not None:
    print(f"\nBest DBSCAN: eps={best_eps}, silhouette={best_silhouette:.4f}")

    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        color = 'gray' if label == -1 else None
        plt.scatter(label_points[:, 0], label_points[:, 1], c=color, label=f'Cluster {label}' if label != -1 else 'Noise', alpha=0.6)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"DBSCAN Clustering (eps={best_eps})")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("⚠️ DBSCAN did not find a valid clustering configuration with more than one cluster.")

#try2
# Initialize OPTICS
min_samples_values = [2,5, 10, 20, 30]  # Varying min_samples values
best_silhouette = -1
best_min_samples = None
best_labels = None

# Loop over min_samples to test different configurations
for min_samples in min_samples_values:
    optics = OPTICS(min_samples=min_samples, metric='minkowski',xi=0.03, min_cluster_size=0.1)
    labels = optics.fit_predict(X_normalized)  # X_normalized: your scaled trajectory vectors
    
    # Skip silhouette score calculation if all points are noise or in one cluster
    if len(set(labels)) > 1:  # Ensure more than one cluster exists
        silhouette = silhouette_score(X_normalized, labels)
        print(f"OPTICS: min_samples={min_samples}, silhouette={silhouette}")
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_min_samples = min_samples
            best_labels = labels

if best_labels is not None:
    print(f"Best OPTICS: min_samples={best_min_samples}, silhouette={best_silhouette}")

    # Visualization of OPTICS results
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_labels)
    for label in unique_labels:
        label_points = X_normalized[best_labels == label]
        if label == -1:  # Noise points
            plt.scatter(label_points[:, 0], label_points[:, 1], c='gray', label='Noise', alpha=0.5)
        else:
            plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Cluster {label}')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"OPTICS Clustering (min_samples={best_min_samples})")
    plt.legend()
    plt.show()
else:
    print("OPTICS could not find any meaningful clusters with the given parameters.")

# Optional: Extract Reachability Plot
reachability = optics.reachability_[optics.ordering_]
plt.figure(figsize=(10, 6))
plt.plot(reachability)
plt.title("OPTICS Reachability Plot")
plt.xlabel("Sample Index")
plt.ylabel("Reachability Distance")
plt.show()
