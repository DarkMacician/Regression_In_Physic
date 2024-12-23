import matplotlib.pyplot as plt

# Function to determine the cross product
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# Andrew's Monotone Chain Algorithm
def monotone_chain(points):
    points = sorted(points)  # Step 1: Sort the points

    # Step 2: Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Step 3: Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it's repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]

# Sample set of points (replace with your own coordinates)
points = [(1, 1), (2, 5), (3, 3), (5, 5), (3, 2), (4, 3), (6, 1)]

# Compute the convex hull
hull = monotone_chain(points)

# Convert hull to a list of x and y coordinates
hull_x, hull_y = zip(*hull)

# Plot the points and the convex hull
plt.figure(figsize=(6, 6))
plt.scatter(*zip(*points), color='blue', label='Points')  # Scatter plot for points
plt.plot(hull_x + (hull_x[0],), hull_y + (hull_y[0],), color='red', label='Convex Hull')  # Convex hull border
plt.fill(hull_x + (hull_x[0],), hull_y + (hull_y[0],), color='orange', alpha=0.3)  # Fill the convex hull
plt.title('Convex Hull of Points')
plt.legend()
plt.show()
