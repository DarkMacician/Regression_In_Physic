from folium import Polygon
from shapely.geometry import Point

# Example coordinate to check (replace with your coordinate)
coordinate = (106.3, 10.8)  # Example coordinate (Longitude, Latitude)

# Step 1: Define the main area of interest (AOI) as a Polygon
main_area_polygon = Polygon([
    (106.2, 10.7), (106.5, 10.7),
    (106.5, 10.9), (106.2, 10.9),
    (106.2, 10.7)
])

# Step 2: Create a Shapely Point object from the coordinate
point = Point(coordinate)

# Step 3: Check if the point is inside the polygon
if main_area_polygon.contains(point):
    print(f"The point {coordinate} is inside the area.")
else:
    print(f"The point {coordinate} is outside the area.")
