# Use bokeh to allow user interactivity
# All units in arcseconds
# X is dispersion direction (along short shutter length)
# Y is cross-dispersion direction (along slitlet)
#
# Run this as a Bokeh SERVER app (not `python show_ifs.py`), e.g.:
#   bokeh serve --show show_ifs_throughput_interactive.py

from astropy.io import fits

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# ---- Fixed shutter geometry -------------------------------------------------
shutter_width = 0.2
shutter_height = 0.46
shutter_gap = 0.07
shutter_x = 0
shutter_y = 0

# Slitlet
slitlet_shutter = 5
# Side step
mosaic_step_size = 0.2
mosaic_step_number = 5

# Wavelength selection
wavelength_index = 5

# ---- Create MSA slitlet throughput from pathloss file -----------------------

# TODO: handle nicely missing file
pathloss_file = 'jwst_nirspec_pathloss_0010.fits'
srctype = 'PS'
aperture = 'MOS1x1'
scale = [0.27, 0.53, 1e6]  # arcsec/shutter, arcsec/shutter, µm/m

# TODO: limit pixel_scale to non arbitrary small values
pixel_scale = 0.005  # arcsec/pixel

# Read pathloss from calibration reference file.
with fits.open(pathloss_file) as hdulist:
    for hdu in hdulist:
        head = hdu.header
        try:
            if head['EXTNAME'] == srctype and head['APERTURE'] == aperture:
                cube = hdu.data
                break
        except KeyError:
            pass

# Construct world coordinate axes for cube in sky and wavelength units.
axes = []
for i in range(1, head['NAXIS'] + 1):
    n = head[f'NAXIS{i}']
    crpix = head[f'CRPIX{i}']
    crval = head[f'CRVAL{i}']
    cdelt = head[f'CDELT{i}']
    index = np.arange(1, n + 1)
    axis = scale[i - 1] * (crval + (index - crpix) * cdelt)
    axes.append(axis)
xcube, ycube, wcube = axes
wcube_str = [f'{w:.2f}' for w in wcube]

# TODO: handle possible NaN failure in FFT nicely
# New regular grid
X_new, Y_new = np.meshgrid(
    np.arange(xcube.min(), xcube.max(), pixel_scale),
    np.arange(ycube.min(), ycube.max(), pixel_scale)
)
points = np.column_stack([Y_new.ravel(), X_new.ravel()])
# Output resample shutter, at each wavelength
shutter_resampled = np.empty((len(wcube), *Y_new.shape))
for i, w in enumerate(wcube):

    # Resample Throughput map on regular finer grid for each wavelength
    interp = RegularGridInterpolator(
        (ycube, xcube),   # order must match z dimensions
        cube[i, :, :],
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # Shutter throughput resampled
    shutter_resampled[i, :, :] = interp(points).reshape(Y_new.shape)


# ----- Functions needed to compute the display -----------------------------

def _fft_convolve_same(a, b):
    """Computes the 2D linear convolution of `a` and `b`, cropped to the shape
    of `a` with `b` centered on each output point.

    Equivalent to ``scipy.signal.fftconvolve(a, b, mode='same')`` but avoids
    an extra scipy dependency since we only need this one operation.

    Parameters
    ----------
    a : numpy.ndarray, shape (ny, nx)
        The larger "signal" array (e.g. the shutter position grid).
    b : numpy.ndarray, shape (kh, kw)
        The smaller "kernel" array to convolve with ``a``. Assumed to be
        centered at ``(kh // 2, kw // 2)``.

    Returns
    -------
    numpy.ndarray, shape (ny, nx)
        The convolution of `a` and `b`, same shape as `a`.
    """
    s1 = np.array(a.shape)
    s2 = np.array(b.shape)
    size = s1 + s2 - 1
    fa = np.fft.rfft2(a, size)
    fb = np.fft.rfft2(b, size)
    conv = np.fft.irfft2(fa * fb, size)
    start = (s2 - 1) // 2
    end = start + s1
    return conv[start[0]:end[0], start[1]:end[1]]


def compute_throughput_map(mosaic_xs, mosaic_ys, kernel, pixel_scale):
    """Stack shifted copies of the per-shutter throughput kernel at every
    mosaic shutter center and sum them.

    Values in the returned map can thus exceed 1 wherever multiple exposures
    overlap the same point on the sky.

    Implemented as:
    1. Since user coordinates might be more precise than the pixel scale,
    we split each shutter into four, located at the 4 nearest pixels, with
    weights equal to how close the real coordinates are.
    2. Convolve that grid with the kernel via FFT, which gives an exact sum of
    shifted kernel copies which is more efficient than looping per shutter and
    adding the kernel into a big array by hand.

    Parameters
    ----------
    mosaic_xs : sequence of float
        X (dispersion-direction, arcsec) center position of every shutter in
        the mosaic
    mosaic_ys : sequence of float
        Y (cross-dispersion direction, arcsec) center position of every shutter
        in the mosaic. Must be the same length as `mosaic_xs`.
    kernel : numpy.ndarray, shape (kh, kw)
        The single-shutter throughput profile sampled at `pixel_scale` arcsec
        per pixel with the shutter's optical center at the array's midpoint.
    pixel_scale : float
        Sampling of `kernel` and of the output map, in arcsec/pixel. Must
        match the scale at which the `kernel` is sampled.

    Returns
    -------
    throughput_map : numpy.ndarray, shape (ny, nx)
        The summed throughput at every grid point.
    x0 : float
        X (arcsec) coordinate of the map's lower-left corner.
    y0 : float
        Y (arcsec) coordinate of the map's lower-left corner.
    width : float
        Extent of the map along X (arcsec).
    height : float
        Extent of the map along Y (arcsec).
    """

    # All shutter centers
    xs = np.asarray(mosaic_xs, dtype=float)
    ys = np.asarray(mosaic_ys, dtype=float)
    # 1-shutter kernel size
    kh, kw = kernel.shape

    # Pad the extent by half the kernel size on each side so kernel not clipped
    pad_x = kw * pixel_scale / 2
    pad_y = kh * pixel_scale / 2
    x_min = xs.min() - pad_x
    x_max = xs.max() + pad_x
    y_min = ys.min() - pad_y
    y_max = ys.max() + pad_y

    # Dimension of the map
    nx = max(kw, int(np.ceil((x_max - x_min) / pixel_scale)))
    ny = max(kh, int(np.ceil((y_max - y_min) / pixel_scale)))

    # Position, in pixel, of each shutter in the sampled grid as a float
    fx = (xs - x_min) / pixel_scale
    fy = (ys - y_min) / pixel_scale
    # - as an integer, limited to the size of the array
    #   (avoiding last column/row because of the 4-split below)
    ix0 = np.clip(np.floor(fx).astype(int), a_min=0, a_max=nx - 2)
    iy0 = np.clip(np.floor(fy).astype(int), a_min=0, a_max=ny - 2)
    # - fractional pixel residual to be used as weights
    wx = fx - ix0
    wy = fy - iy0
    # - splitting each shutter in 4, with weights, onto sampled grid
    position = np.zeros((ny, nx))
    np.add.at(position, (iy0, ix0), (1 - wx) * (1 - wy))
    np.add.at(position, (iy0, ix0 + 1), wx * (1 - wy))
    np.add.at(position, (iy0 + 1, ix0), (1 - wx) * wy)
    np.add.at(position, (iy0 + 1, ix0 + 1), wx * wy)

    # Combining all shutters into one throughput map
    throughput_map = _fft_convolve_same(position, kernel)
    # Clean up near-zero values
    throughput_map[np.abs(throughput_map) < 1e-9] = 0.0

    return throughput_map, x_min, y_min, x_max - x_min, y_max - y_min


def compute_and_update(wavelength_index,
                       slitlet_shutter,
                       mosaic_step_size,
                       mosaic_step_number,
                       dither_xs, dither_ys,
                       return_maps=False):
    """Recompute the full shutter/dither/mosaic geometry and push every
     derived quantity into the plot's data sources and annotations.

     This is the function that updates what is drawn: the shutter rectangles,
     the dither-center crosses, the stacked throughput heatmap, and the red
     mosaic-step arrows.

     Parameters
     ----------
     slitlet_shutter : int
         Number of shutters in a single slitlet (>= 1).
     mosaic_step_size : float
         Spacing between consecutive mosaic side-step along X, in arcsec (>0).
     mosaic_step_number : int
         Number of mosaic side steps along X (>= 1).
     dither_xs : sequence of float
         X offsets (arcsec) of the active dither pattern, applied on top of
         every slitlet shutter position.
     dither_ys : sequence of float
         Y offsets (arcsec) of the active dither pattern, applied on top of
         every slitlet shutter position. Must be the same length as
         `dither_xs`.

     Returns
     -------
     stdev: the standard deviation in the core of the throughput map
     """

    # Slitlet's shutters positions: given by number of shutter in slitlet
    # - handle odd/even number of shutters
    odd_shutters = bool(slitlet_shutter % 2)
    min_shutter = -(slitlet_shutter//2) if odd_shutters else -slitlet_shutter / 2
    max_shutter = slitlet_shutter//2 + 1 if odd_shutters else slitlet_shutter / 2
    # - positions of shutters
    slitlet_xs = [
        shutter_x
        for _ in np.arange(min_shutter, max_shutter)
    ]
    slitlet_ys = [
        shutter_y + i * (shutter_height + shutter_gap)
        for i in np.arange(min_shutter, max_shutter)
    ]

    # All dither positions for the slitlet: given by dither pattern table
    all_dither_xs = [ssx + dx for ssx in slitlet_xs for dx in dither_xs]
    all_dither_ys = [ssy + dy for ssy in slitlet_ys for dy in dither_ys]

    # Dither positions for first shutter only
    ori_dither_xs = [shutter_x + dx for dx in dither_xs]
    ori_dither_ys = [shutter_y + dy for dy in dither_ys]

    # All mosaic positions
    # - ensure that mosaic is centered
    min_step = -(mosaic_step_number - 1) / 2
    max_step = (mosaic_step_number - 1) / 2 + 1
    # - positions of shutters
    all_mosaic_xs = [
        i * mosaic_step_size + dx
        for i in np.arange(min_step, max_step)
        for dx in all_dither_xs
    ]
    all_mosaic_ys = [
        dy
        for _ in np.arange(min_step, max_step)
        for dy in all_dither_ys
    ]

    # Mosaic positions for first shutter only
    ori_mosaic_xs = [
        i * mosaic_step_size + shutter_x
        for i in np.arange(min_step, max_step)
    ]
    ori_mosaic_ys = [
        shutter_y
        for _ in np.arange(min_step, max_step)
    ]

    # Stacked throughput heatmap
    throughput_map, x0, y0, w, h = compute_throughput_map(
        all_mosaic_xs, all_mosaic_ys,
        shutter_resampled[wavelength_index, :, :], pixel_scale)

    # print(all_mosaic_xs)
    # print(all_mosaic_ys)

    # Standard deviation over core
    x_from = int(- x0 / pixel_scale)
    y_from = int(- y0 / pixel_scale)
    x_to = int((w + x0) / pixel_scale)
    y_to = int((h + y0) / pixel_scale)
    med = np.median(throughput_map[y_from:y_to, x_from:x_to])
    stdev = np.std(throughput_map[y_from:y_to, x_from:x_to])
    pv = (np.max(throughput_map[y_from:y_to, x_from:x_to])
          - np.min(throughput_map[y_from:y_to, x_from:x_to]))

    # Median vertical and horizontal profiles
    med_horizontal_profile = np.median(throughput_map, axis=0)
    med_vertical_profile = np.median(throughput_map, axis=1)
    x_med = np.median(med_horizontal_profile[x_from:x_to])
    y_med = np.median(med_vertical_profile[y_from:y_to])
    x_stdev = np.std(med_horizontal_profile[x_from:x_to])
    y_stdev = np.std(med_vertical_profile[y_from:y_to])
    x_pv = (np.max(med_horizontal_profile[x_from:x_to])
            - np.min(med_horizontal_profile[x_from:x_to]))
    y_pv = (np.max(med_vertical_profile[y_from:y_to])
            - np.min(med_vertical_profile[y_from:y_to]))

    # Compute ratio metrics
    ratio, x_ratio, y_ratio = stdev / med, x_stdev / x_med, y_stdev / y_med

    if return_maps:
        return (ratio, x_ratio, y_ratio), throughput_map, med_horizontal_profile, med_vertical_profile
    return ratio, x_ratio, y_ratio


def wrapper_function(dither_table):
    """
    Dither table is list like this:
    [ x1, x2, ... y1, y2, ... ]
    """
    n_dither = int(len(dither_table)/2)
    dither_xs = [0,] + list(dither_table[:n_dither])
    dither_ys = [0,] + list(dither_table[n_dither:])

    res, x, y = compute_and_update(
        wavelength_index,
        slitlet_shutter, mosaic_step_size, mosaic_step_number,
        dither_xs, dither_ys)

    return res


def save_diagnostic_plots(throughput_map, med_horizontal_profile, med_vertical_profile,
                           map_filename='best_dither_pattern_map.pdf',
                           profile_filename='best_dither_pattern.pdf'):
    # Save plots
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.tight_layout(pad=3)
    ax.imshow(throughput_map, cmap='viridis', origin='lower')
    plt.savefig(map_filename, bbox_inches='tight')
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=3)
    ax1.plot(med_horizontal_profile)
    ax2.plot(med_vertical_profile)
    plt.savefig(profile_filename, bbox_inches='tight')
    plt.close(fig)


# ----- Execute code ----------------------------------------------------------

# initial guess as [ x0, x1, x2, ... y0, y1, y2, ... ]
x0 = [0.1,    0.05,  0.1,   0.05,
      0.1325, 0.05, 0.265, 0.3975]

n_dither = int(len(x0) / 2)

bounds = [(0.001, 0.2)] * n_dither + [(0.001, 0.4)] * n_dither
print(bounds)

# result = minimize(wrapper_function, x0,
#                   bounds=bounds,
#                   method='L-BFGS-B')
result = differential_evolution(wrapper_function, bounds, x0=x0)


# Save results into plots
best_dither_xs = [0,] + list(result.x[:n_dither])
best_dither_ys = [0,] + list(result.x[n_dither:])

_, throughput_map, med_h, med_v = compute_and_update(
    wavelength_index,
    slitlet_shutter, mosaic_step_size, mosaic_step_number,
    best_dither_xs, best_dither_ys, return_maps=True)

save_diagnostic_plots(throughput_map, med_h, med_v)


print("Optimal parameters (X):", [0,] + result.x[:int(len(x0)/2)])
print("Optimal parameters (Y):", [0,] + result.x[int(len(x0)/2):])
print("Minimum value:", result.fun)
