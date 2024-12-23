import matplotlib.pyplot as plt
import numpy as np

# Define the lines
x1 = np.linspace(0, 40, 100)  # For y = 80 - 2x and y = 5 + 3x
y1 = 80 - 2 * x1  # y = 80 - 2x
y2 = 5 + 3 * x1  # y = 5 + 3x

x3 = np.linspace(13, 18, 100)  # For y = 44 (13 <= x <= 18)
y3 = np.full_like(x3, 44)  # y = 44

x4 = np.linspace(35/3, 20, 100)  # For y = 40 (35/3 <= x <= 20)
y4 = np.full_like(x4, 40)  # y = 40

# Define vertical lines with specific y-ranges
y5 = np.linspace(0, 40, 100)
x5 = np.full_like(y5, 35/3)  # x = 35/3 (0 <= y <= 40)

y6 = np.linspace(0, 40, 100)
x6 = np.full_like(y6, 20)  # x = 20 (0 <= y <= 40)

y7 = np.linspace(0, 44, 100)
x7 = np.full_like(y7, 18)  # x = 18 (0 <= y <= 44)

y8 = np.linspace(0, 44, 100)
x8 = np.full_like(y8, 13)  # x = 13 (0 <= y <= 44)

# Plotting the lines
plt.figure(figsize=(8, 8))

# Plot all lines and segments
plt.plot(x1, y1, label="y = 80 - 2x", color="blue", linewidth=1.5)
plt.plot(x1, y2, label="y = 5 + 3x", color="orange", linewidth=1.5)
plt.plot(x3, y3, label="y = 44 (13 <= x <= 18)", color="green", linewidth=1.5)
plt.plot(x4, y4, label="y = 40 (35/3 <= x <= 20)", color="red", linewidth=1.5)
plt.plot(x5, y5, label="x = 35/3 (0 <= y <= 40)", linestyle="--", color="purple", linewidth=1.5)
plt.plot(x6, y6, label="x = 20 (0 <= y <= 40)", linestyle="--", color="brown", linewidth=1.5)
plt.plot(x7, y7, label="x = 18 (0 <= y <= 44)", linestyle="--", color="pink", linewidth=1.5)
plt.plot(x8, y8, label="x = 13 (0 <= y <= 44)", linestyle="--", color="gray", linewidth=1.5)

# Customize grid and axes
plt.grid(color="lightgray", linestyle="--", linewidth=0.5)  # Desmos-like grid
plt.axhline(0, color="black", linewidth=1)  # x-axis at y=0
plt.axvline(0, color="black", linewidth=1)  # y-axis at x=0
plt.gca().spines["top"].set_visible(False)  # Remove top spine
plt.gca().spines["right"].set_visible(False)  # Remove right spine

# Set axis limits to zoom in
plt.xlim(0, 40)  # Limit horizontal axis to focus area
plt.ylim(0, 80)  # Limit vertical axis to focus area

# Labels and title
plt.xlabel("Q", fontsize=12)
plt.ylabel("P", fontsize=12)
#plt.title("Zoomed-In Graph of Given Lines and Segments", fontsize=14)

# Add legend outside the plot
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)

plt.tight_layout()
plt.show()
