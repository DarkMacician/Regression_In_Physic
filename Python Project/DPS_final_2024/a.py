import osmnx as ox
import geopandas as gpd
import folium
from shapely.geometry import Polygon

# Specify the path to your OSM file
osm_file = "planet_106.349_10.76_9a1e0f2e.osm"

# Step 1: Load the boundaries from the OSM file
gdf = ox.features_from_xml(osm_file, tags={"boundary": True})

# Step 2: Define your main area of interest (AOI)
# Replace with your specific boundary polygon
main_area_polygon = Polygon([
    (106.2, 10.7), (106.5, 10.7),
    (106.5, 10.9), (106.2, 10.9),
    (106.2, 10.7)
])
main_area_gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[main_area_polygon])

# Step 3: Keep only boundaries intersecting with the main area
filtered_gdf = gdf[gdf.intersects(main_area_polygon)]

# Step 4: Create an interactive map with Folium
# Get the centroid of the AOI to center the map
centroid = main_area_gdf.geometry[0].centroid
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

# Step 5: Add the AOI to the map
folium.GeoJson(
    main_area_gdf,
    style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0.1},
    name="Area of Interest"
).add_to(m)

# Step 6: Add the filtered boundaries to the map
folium.GeoJson(
    filtered_gdf,
    style_function=lambda x: {"color": "blue", "weight": 1.5},
    name="Filtered Boundaries"
).add_to(m)

# Step 7: Add a layer control to toggle visibility of layers
folium.LayerControl().add_to(m)

# Step 8: Save and display the map
m.save("filtered_boundaries_map.html")
print("Map saved to 'filtered_boundaries_map.html'. Open this file in a browser to view the map.")
