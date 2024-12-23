import osmnx as ox
import networkx as nx
import folium

# Step 1: Specify the OSM file
osm_file = "planet_106.349_10.76_9a1e0f2e.osm"

# Step 2: Load the graph from the OSM file
G = ox.graph_from_xml(osm_file)

# Step 3: Filter the graph by network type (e.g., drive, walk)
# For example, to get a driving network, you can filter by the 'highway' attribute
# We keep only the edges that are 'motorway', 'trunk', 'primary', 'secondary', etc.
drive_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service']

# We need to iterate through the edges and check for the 'highway' attribute
edges_to_keep = []
for u, v, key, data in G.edges(keys=True, data=True):
    if 'highway' in data and data['highway'] in drive_types:
        edges_to_keep.append((u, v, key))

# Create a subgraph with the filtered edges
G_filtered = G.edge_subgraph(edges_to_keep)

# Step 4: Define start and end points (latitude, longitude)
point_a = (10.762622, 106.660172)  # Example coordinates
point_b = (10.775658, 106.700424)

# Step 5: Find nearest nodes to the specified points
node_a = ox.nearest_nodes(G_filtered, X=point_a[1], Y=point_a[0])
node_b = ox.nearest_nodes(G_filtered, X=point_b[1], Y=point_b[0])

# Step 6: Calculate the shortest path
shortest_path = nx.shortest_path(G_filtered, source=node_a, target=node_b, weight="length")

# Step 7: Plot the shortest path on a map using folium
# Create a folium map centered at the midpoint between point_a and point_b
midpoint = ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)
m = folium.Map(location=midpoint, zoom_start=14)

# Add start and end markers
folium.Marker(location=point_a, popup="Start Point", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(location=point_b, popup="End Point", icon=folium.Icon(color="red")).add_to(m)

# Extract path coordinates
path_coords = [(G_filtered.nodes[node]["y"], G_filtered.nodes[node]["x"]) for node in shortest_path]

# Add the shortest path as a PolyLine
folium.PolyLine(locations=path_coords, color="blue", weight=5, opacity=0.7).add_to(m)

# Save the map to an HTML file
map_file = "shortest_path_map.html"
m.save(map_file)
print(f"Map with shortest path saved to {map_file}")