import osmnx as ox
import folium
from shapely.geometry import Polygon
import geopandas as gpd
import json

# Specify the path to your OSM file
osm_file = "planet_106.349_10.76_9a1e0f2e.osm"

# Step 1: Load the boundaries from the OSM file
gdf = ox.features_from_xml(osm_file, tags={"boundary": True})

# Step 2: Define your main area of interest (AOI) (example)
main_area_polygon = Polygon([
    (106.2, 10.7), (106.5, 10.7),
    (106.5, 10.9), (106.2, 10.9),
    (106.2, 10.7)
])
main_area_gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[main_area_polygon])

# Step 3: Query OSM for amenities (e.g., schools, hospitals)
schools = ox.features_from_xml(osm_file, tags={"amenity": "school"})
hospitals = ox.features_from_xml(osm_file, tags={"amenity": "hospital"})
target_name = "Bệnh viện Đa khoa khu vực Hậu Nghĩa"

# Search for schools or hospitals with that name in their attributes
matching_schools = schools[schools["name"].str.contains(target_name, case=False, na=False)]
matching_hospitals = hospitals[hospitals["name"].str.contains(target_name, case=False, na=False)]

# Step 5: Plot the map and add markers
# Create a Folium map centered on the area of interest
centroid = main_area_gdf.geometry[0].centroid
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

# Add filtered boundaries (convert to GeoJSON and add to Folium)
geojson_boundaries = gdf.to_crs(epsg=4326).__geo_interface__  # Convert to GeoJSON
folium.GeoJson(geojson_boundaries, name="Boundaries", style_function=lambda x: {"color": "blue", "weight": 1.5}).add_to(m)

# Add markers for the matching locations
for _, location in matching_schools.iterrows():
    location_coords = [location.geometry.centroid.y, location.geometry.centroid.x]
    folium.Marker(location_coords, popup=location["name"], icon=folium.Icon(color="green")).add_to(m)

for _, location in matching_hospitals.iterrows():
    location_coords = [location.geometry.centroid.y, location.geometry.centroid.x]
    folium.Marker(location_coords, popup=location["name"], icon=folium.Icon(color="red")).add_to(m)

# Step 6: Add a layer control and save the map
folium.LayerControl().add_to(m)
m.save("map_with_target_location_marker.html")

print("Map with target location marker saved to 'map_with_target_location_marker.html'. Open this file in a browser to view the map.")