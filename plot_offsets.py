import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from find_best_dither_pattern import *
from itertools import permutations
from matplotlib.patches import Rectangle


# Wavelengths
w = [0.60000002, 0.87000003, 1.14000004, 1.41000005, 1.68000006,
     1.95000007, 2.22000008, 2.49000009, 2.7600001, 3.03000011,
     3.30000012, 3.57000013, 3.84000014, 4.11000015, 4.38000015,
     4.65000016, 4.92000017, 5.19000018, 5.46000019, 5.7300002,
     6.00000021]

# Offsets found for the monochromatic cases
x0 = [(-0.100, -0.032, -0.062, +0.030), (-0.028, +0.065, -0.068, +0.032),
     (-0.033, +0.027, -0.063, +0.067), (+0.031, -0.029, -0.069, +0.065),
     (+0.068, +0.031, -0.069, -0.030), (-0.030, +0.037, +0.060, +0.104),
     (-0.061, +0.037, -0.031, -0.100), (-0.100, -0.061, -0.031, +0.037),
     (+0.080, +0.062, +0.133, +0.020), (-0.111, -0.051, -0.130, -0.070),
     (-0.062, +0.076, +0.018, -0.043), (-0.069, +0.061, +0.041, -0.018),
     (+0.043, -0.020, +0.061, +0.119), (+0.044, +0.120, -0.019, +0.062),
     (-0.012, +0.051, +0.126, +0.068), (-0.021, +0.048, +0.069, +0.119),
     (-0.070, +0.050, -0.090, -0.022), (+0.031, +0.071, -0.038, -0.069),
     (-0.020, +0.048, +0.120, +0.069), (+0.068, -0.071, +0.049, -0.021),
     (-0.042, -0.071, +0.077, +0.029)]
y0 = [(-0.213, -0.106, +0.106, +0.210), (-0.212, -0.105, +0.107, +0.212),
     (-0.213, -0.105, +0.106, +0.213), (-0.212, -0.105, +0.107, +0.214),
     (-0.212, -0.106, +0.106, +0.212), (-0.212, -0.105, +0.106, +0.212),
     (-0.212, -0.105, +0.107, +0.214), (-0.213, -0.105, +0.107, +0.213),
     (-0.213, -0.105, +0.106, +0.213), (-0.211, -0.104, +0.107, +0.212),
     (-0.212, -0.105, +0.107, +0.212), (-0.212, -0.105, +0.107, +0.212),
     (-0.213, -0.105, +0.107, +0.213), (-0.212, -0.105, +0.107, +0.211),
     (-0.212, -0.105, +0.107, +0.211), (-0.212, -0.105, +0.109, +0.215),
     (-0.211, -0.105, +0.106, +0.211), (-0.210, -0.102, +0.108, +0.215),
     (-0.210, -0.106, +0.107, +0.213), (-0.212, -0.105, +0.108, +0.215),
     (-0.211, -0.103, +0.107, +0.213)]

# Convert to arrays
w = np.array(w)
x = np.array(x0)  # shape (21, 4)
y = np.array(y0)  # shape (21, 4)

# Create colormap mapping
norm = Normalize(vmin=w.min(), vmax=w.max())
cmap = cm.rainbow

fig, ax = plt.subplots()
for i, wl in enumerate(w):
    color = cmap(norm(wl))
    ax.scatter(x[i], y[i], color=color, s=50)
ax.set_xlabel("X offsets")
ax.set_ylabel("Y offsets")
ax.set_title("X vs Y colored by wavelength")
# Add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Wavelength")
plt.savefig('monochromatic_colored_offsets.pdf')
plt.show()


# Flatten and sorted
xx = [a for t in x0 for a in t]
yy = [a for t in y0 for a in t]
xx = np.sort(xx)
yy = np.sort(yy)
fig, ax = plt.subplots()
ax.scatter(xx, yy, alpha=0.3, s=50)
ax.set_xlabel("X offsets")
ax.set_ylabel("Y offsets")
ax.set_title("X vs Y, both sorted")
plt.savefig('monochromatic_offsets.pdf')
plt.show()
# print stats for each dither position
for i in range(len(x0[0])):
    sel_x = xx[i * len(x0):(i + 1) * len(x0)]
    sel_y = yy[i * len(x0):(i + 1) * len(x0)]
    print("(X) Mean, median, stdev:",
          np.mean(sel_x), np.median(sel_x), np.std(sel_x))
    print("(Y) Mean, median, stdev:",
          np.mean(sel_y), np.median(sel_y), np.std(sel_y))

# Distribution of X values
fig, ax = plt.subplots()
ax.hist(xx, bins=30)
ax.set_ylabel("X offsets")
ax.set_title("X distribution")
plt.savefig('x_histogram.pdf')
plt.show()
fig, ax = plt.subplots()
ax.plot(xx, 'C0o', alpha=0.3)
ax.set_ylabel("X offsets")
ax.set_title("X distribution")
plt.savefig('x_offsets.pdf')
plt.show()


# Test different order in the offsets
x0 = [-0.062, 0.076, 0.018, -0.043,
      -0.21, -0.11, 0.11, 0.21]
m = []
# for p in permutations(x0[:4]):
#      x = list(p) + x0[4:]
#      m.append(wrapper_function(x))
# print(f'{(max(m)/min(m) - 1) * 100:.4f} %')
# ===> maximum variation due to order is 3%


# Show subpixel maps

# 6-dither pattern (polychromatic)
x = [0, 0.075, -0.062, -0.062, 0.067, 0.009]
y = [0, -0.264, -0.176, 0.087, 0.089, 0.178]

# 8-dither pattern (polychromatic)
x = [0, 0.029, 0.064, -0.070, -0.070, -0.039, -0.003, 0.070]
y = [0, -0.260, -0.197, -0.129, -0.063, 0.069, 0.134, 0.200]
# 9-dither pattern (polychromatic)
x = [0, 0.105, -0.022, 0.050, -0.041, 0.035, -0.041, 0.096, 0.026]
y = [0, -0.231, -0.180, -0.117, -0.060, 0.059, 0.117, 0.178, 0.231]
# 10-dither pattern (polychromatic)
x = [0, 0.091, -0.041, 0.042, -0.067, 0.073, -0.008, 0.013, -0.060, 0.034]
y = [0, -0.238, -0.228, -0.148, -0.116, -0.071, 0.059, 0.109, 0.162, 0.219]

# 5-dither pattern (poly * distances)
x = [0, 0.064, -0.068, 0.072, -0.068]
y = [0, -0.218, -0.107, 0.104, 0.213]
# 5-dither pattern (polychromatic)
x = [0, 0.030, -0.039, -0.069, 0.070]
y = [0, -0.213, -0.105, 0.107, 0.214]

# 7-dither pattern (polychromatic)
x = [0, 0.051, -0.014, -0.078, -0.027, 0.051, -0.079]
y = [0, -0.227, -0.152, -0.077, 0.075, 0.152, 0.227]
# 7-dither pattern (poly * distances)
x = [0, 0.134, -0.129, 0.135, -0.135, -0.128, 0.135]
y = [0, -0.265, -0.263, -0.000, 0.000, 0.264, 0.265]
# 7-dither pattern (poly * sqrt(distances))
x = [0, -0.110, 0.029, -0.110, -0.049, 0.024, -0.041]
y = [0, -0.222, -0.143, -0.068, 0.083, 0.157, 0.238]
# 7-dither pattern (poly * min(distances))
x = [0, 0.027, -0.052, 0.078, 0.077, -0.052, 0.013]
y = [0, -0.226, -0.149, -0.075, 0.077, 0.146, 0.231]


fig, ax = plt.subplots(figsize=(8, 8))
ax.hlines([0, 1], 0, 1, alpha=0.25, color='black')
ax.vlines([0, 1], 0, 1, alpha=0.25, color='black')
ax.set_xlabel("Sub-pixel X position", fontsize=14)
ax.set_ylabel("Sub-pixel Y position", fontsize=14)
for i, j in zip(x, y):
    xx = (i % 0.1) / 0.1
    if xx > 0.5:
        xx -= 1
    yy = (j % 0.1) / 0.1
    if yy > 0.5:
        yy -= 1
    box = Rectangle((xx, yy), 1, 1, alpha=0.1, ec='black')
    ax.add_patch(box)
ax.axis('equal')
ax.set_xlim(-.25, 1.25)
ax.set_ylim(-.25, 1.25)
plt.savefig(f'subpixelcov_{len(x):.0f}dither.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ax.hlines([-0.5, 0.5], -0.5, 0.5, alpha=0.25, color='black')
ax.vlines([-0.5, 0.5], -0.5, 0.5, alpha=0.25, color='black')
ax.set_xlabel("Sub-pixel X position", fontsize=14)
ax.set_ylabel("Sub-pixel Y position", fontsize=14)
# We want to plot the centers
xc, yc = [], []
for i, j in zip(x, y):
    xx = (i % 0.1) / 0.1
    xc.append(xx - 1 if xx > 0.5 else xx)

    yy = (j % 0.1) / 0.1
    yc.append(yy - 1 if yy > 0.5 else yy)

ax.plot(xc, yc, 'kx', label='Dither centers')
ax.axis('equal')
ax.legend()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
plt.savefig(f'subpixel_{len(x):.0f}dither.pdf')
plt.show()


# Test different offsets
# 7-dither pattern (polychromatic)
x0 = [0.051, -0.014, -0.078, -0.027, 0.051, -0.079,
      -0.227, -0.152, -0.077, 0.075, 0.152, 0.227]
wrapper_function(x0)
# 7-dither pattern (poly * min(distances))
x0 = [0.027, -0.052, 0.078, 0.077, -0.052, 0.013,
      -0.226, -0.149, -0.075, 0.077, 0.146, 0.231]
wrapper_function(x0)
# 7-dither pattern (manual)
x0 = [0.025, -0.050, 0.075, 0.075, -0.05, 0.025,
      -0.225, -0.150, -0.075, 0.075, 0.150, 0.225]
wrapper_function(x0)
