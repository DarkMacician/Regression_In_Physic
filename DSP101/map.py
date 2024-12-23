import numpy as np
import matplotlib.pyplot as plt

# Sample data for heights
fathers_heights = np.random.normal(175, 7, 100)
mothers_heights = np.random.normal(165, 6, 100)
children_heights = np.random.normal(170, 8, 100)

# Calculate the required min and max heights
min_father = np.min(fathers_heights)
max_father = np.max(fathers_heights)
min_mother = np.min(mothers_heights)
max_mother = np.max(mothers_heights)

Hmin = max(min_father, min_mother)
Hmax = min(max_father, max_mother)

# Filter data for plotting
filtered_children_heights = children_heights[(children_heights >= Hmin) & (children_heights <= Hmax)]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.scatter(fathers_heights, children_heights, color='blue', label='Fathers vs Children')
plt.scatter(mothers_heights, children_heights, color='orange', label='Mothers vs Children')
plt.axhline(y=Hmin, color='red', linestyle='--', label='Hmin')
plt.axhline(y=Hmax, color='green', linestyle='--', label='Hmax')
plt.title('Children Heights vs Parents Heights')
plt.xlabel('Parents Heights (cm)')
plt.ylabel('Children Heights (cm)')
plt.xlim(Hmin - 5, Hmax + 5)
plt.ylim(Hmin - 5, Hmax + 5)
plt.legend()
plt.grid()
plt.show()
