import osmnx as ox
import folium
from shapely.geometry import Polygon

# Specify the path to your OSM file
osm_file = "planet_106.349_10.76_9a1e0f2e.osm"

# Load features with the 'amenity' tag
tags = {"amenity": True}
gdf = ox.features_from_xml(osm_file, tags=tags)

# Ask the user for the name of the location to find
location_name = input("Enter the name of the location: ")

# Filter the GeoDataFrame for the entered location name
filtered_gdf = gdf[gdf["name"].str.contains(location_name, case=False, na=False)]

if filtered_gdf.empty:
    print(f"No location found with the name '{location_name}'.")
else:
    # Create a Folium map centered around the first matching location
    first_location = filtered_gdf.iloc[0]
    map_center = [first_location.geometry.centroid.y, first_location.geometry.centroid.x]
    m = folium.Map(location=map_center, zoom_start=14)

    # Add markers and fill the area for all matching locations
    for idx, row in filtered_gdf.iterrows():
        if isinstance(row.geometry, Polygon):
            # Add a filled polygon for the area
            folium.Polygon(
                locations=[(point[1], point[0]) for point in row.geometry.exterior.coords],
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.4,
                popup=row.get("name", "Unnamed Location"),
            ).add_to(m)
        else:
            # Add a marker for non-polygon geometries (points, lines, etc.)
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row.get("name", "Unnamed Location"),
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

    # Save the map as an HTML file
    output_file = "location_map.html"
    m.save(output_file)
    print(f"Map has been saved as {output_file}")
