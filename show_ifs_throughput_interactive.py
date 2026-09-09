# Use bokeh to allow user interactivity
# All units in arcseconds
# X is dispersion direction (along short shutter length)
# Y is cross-dispersion direction (along slitlet)
#
# Create or use a Python environment with the Bokeh package installed.
# Run this as a Bokeh SERVER app in the Terminal with:
#   bokeh serve --show show_ifs_throughput_interactive.py

from astropy.io import fits

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    Arrow, VeeHead, ColumnDataSource, Spinner, Div, Select,
    DataTable, TableColumn, NumberEditor, Switch,
    HoverTool, LinearColorMapper, ColorBar,
)
from bokeh.palettes import Viridis256
from bokeh.layouts import column, row

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---- Fixed shutter geometry -------------------------------------------------
shutter_width = 0.2
shutter_height = 0.46
shutter_gap = 0.07
shutter_x = 0
shutter_y = 0

# Default "5-point dither" mimicking what DLaw showed in the meeting.
#    This seeds the editable dither table.
#    It's no longer used directly in the geometry calculations.
default_dither_xs = [0, 0.027, -0.052, 0.078, 0.077, -0.052, 0.013, 0, 0, 0]
default_dither_ys = [0, -0.226, -0.149, -0.075, 0.077, 0.146, 0.231, 0, 0, 0]

# TODO: limit pixel_scale to non arbitrary small values
pixel_scale = 0.01  # arcsec/pixel


# ---- Create MSA slitlet throughput from pathloss file -----------------------

# TODO: handle nicely missing file
pathloss_file = 'jwst_nirspec_pathloss_0010.fits'
srctype = 'PS'
aperture = 'MOS1x1'
scale = [0.27, 0.53, 1e6]  # arcsec/shutter, arcsec/shutter, µm/m

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


# ---- Main Figure and Data Sources -------------------------------------------

# Initiate figure and axes
p = figure(width=800, height=800, match_aspect=True)
p.xaxis.axis_label = "Arcsecond - dispersed direction"
p.yaxis.axis_label = "Arcsecond - cross-dispersed direction"
p.axis.axis_label_text_font_size = "16pt"
p.axis.major_label_text_font_size = "12pt"

# ColumnDataSources that we need to track
# - coverage/throughput map
coverage_source = ColumnDataSource(
    data=dict(image=[], x=[], y=[], dw=[], dh=[])
)
# - outline of all the shutters
mosaic_source = ColumnDataSource(
    data=dict(x=[], y=[])
)
# - center of all the shutters
dither_plot_source = ColumnDataSource(
    data=dict(x=[], y=[])
)
# - outline of the slitlet
slitlet_source = ColumnDataSource(
    data=dict(x=[], y=[]))

# Show heatmap
coverage_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
coverage_renderer = p.image(
    source=coverage_source, image='image', x='x', y='y', dw='dw', dh='dh',
    color_mapper=coverage_mapper)

# Add colorbar
color_bar = ColorBar(color_mapper=coverage_mapper, title="Throughput",
                     label_standoff=18)
color_bar.major_label_text_font_size = "12pt"
color_bar.title_text_font_size = "12pt"
p.add_layout(color_bar, 'right')

# Display throughput where the mouse hovers
hover = HoverTool(
    renderers=[coverage_renderer],
    tooltips=[("Throughput", "@image{0.00}")],
    mode='mouse')
p.add_tools(hover)

# Show all shutters outline
aper_renderer = p.rect(source=mosaic_source, x='x', y='y',
                       width=shutter_width, height=shutter_height,
                       color=None, alpha=0.25, line_color='black')
# Show dither pattern centers for first shutter only
dith_renderer = p.scatter(source=dither_plot_source, x='x', y='y',
                          marker='x', color='black', size=10)
# Show first slitlets outline darker
slitlet_renderer = p.rect(source=slitlet_source, x='x', y='y',
                          width=shutter_width, height=shutter_height,
                          fill_color=None, line_color='black')
# Show arrowheads for the side-steps
vh = VeeHead(size=15, fill_color='black')
arrow_renderers = []  # track arrows so we can remove/replace them


# ---- Side Figures and Data Sources ------------------------------------------

# Initiate figure and axes - histogram of all throughput values
p_hist = figure(width=600, height=400, title='Throughput Histogram')
p_hist.xaxis.axis_label = "Throughput"
p_hist.yaxis.axis_label = "Occurence"
# Initiate figure and axes - median vertical profile
p_ycut = figure(width=450, height=350, title='Throughput Median Profile')
p_ycut.xaxis.axis_label = "Arcsecond - cross-dispersed direction"
p_ycut.yaxis.axis_label = "Throughput"
# Initiate figure and axes - median horizontal profile
p_xcut = figure(width=450, height=350, title='Throughput Median Profile')
p_xcut.xaxis.axis_label = "Arcsecond - dispersed direction"
p_xcut.yaxis.axis_label = "Throughput"

# Data sources
hist_source = ColumnDataSource(
    data=dict(left=[], right=[], top=[])
)
xcut_source = ColumnDataSource(
    data=dict(x=[], value=[])
)
ycut_source = ColumnDataSource(
    data=dict(x=[], value=[])
)

# Plot histogram
p_hist.quad(left='left', right='right', top='top', bottom=0,
            source=hist_source)
# Plot cuts
p_xcut.line(x='x', y='value', source=xcut_source)
p_ycut.line(x='x', y='value', source=ycut_source)


# ----- Functions needed to recompute the display -----------------------------

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
                       throughput_switch,
                       slitlet_shutter,
                       mosaic_step_size,
                       mosaic_step_number,
                       dither_xs, dither_ys,
                       rect_renderer,
                       slitlet_renderer,
                       dith_renderer):
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
     None
     """

    # TODO: decide if number of shutters will always be odd

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

    # Push new data into sources (triggers redraw of rectangles and crosses)
    mosaic_source.data = dict(
        x=all_mosaic_xs, y=all_mosaic_ys)
    dither_plot_source.data = dict(
        x=ori_dither_xs, y=ori_dither_ys)
    slitlet_source.data = dict(
        x=slitlet_xs, y=slitlet_ys)

    # Stacked throughput heatmap
    throughput_map, x0, y0, w, h = compute_throughput_map(
        all_mosaic_xs, all_mosaic_ys,
        shutter_resampled[wavelength_index, :, :], pixel_scale)
    # Save peak throughput
    coverage_mapper.high = max(float(throughput_map.max()), 1e-6)
    # Push new data into source
    coverage_source.data = dict(
        image=[throughput_map], x=[x0], y=[y0], dw=[w], dh=[h]
    )

    # Rebuild the mosaic arrows (their count depends on mosaic_step_number)
    if arrow_renderers:
        p.center = [c for c in p.center if c not in arrow_renderers]
    arrow_renderers.clear()
    for i in range(mosaic_step_number - 1):
        arrow = Arrow(x_start=ori_mosaic_xs[i], y_start=ori_mosaic_ys[i],
                      x_end=ori_mosaic_xs[i + 1], y_end=ori_mosaic_ys[i + 1],
                      end=vh, line_color='black')
        p.add_layout(arrow)
        arrow_renderers.append(arrow)

    # Turn things ON/OFF
    coverage_renderer.visible = throughput_switch
    color_bar.visible = throughput_switch
    rect_renderer.glyph.line_alpha = 0.25 if throughput_switch else 0.5
    dith_renderer.glyph.line_color = 'black' if throughput_switch else 'red'
    slitlet_renderer.glyph.line_color = 'black' if throughput_switch else 'red'
    slitlet_renderer.glyph.line_width = 1 if throughput_switch else 2
    for arrow in arrow_renderers:
        arrow.visible = throughput_switch

    # Histogram of values in heatmap
    throughput_hist = np.histogram(throughput_map.ravel(), bins=30)
    hist_source.data = dict(
        left=throughput_hist[1][:-1],
        right=throughput_hist[1][1:],
        top=throughput_hist[0])
    # Median vertical and horizontal profiles
    med_horizontal_profile = np.median(throughput_map, axis=0)
    med_vertical_profile = np.median(throughput_map, axis=1)
    xcut_source.data = dict(
        x=np.arange(x0, x0+w, pixel_scale),
        value=med_horizontal_profile)
    ycut_source.data = dict(
        x=np.arange(y0, y0+h, pixel_scale),
        value=med_vertical_profile)


# ---- Input widgets ----------------------------------------------------------

slitlet_shutter_input = Spinner(
    title="Shutters per slitlet (integer \u2265 1)",
    low=1, step=1, value=3, width=300)

mosaic_step_size_input = Spinner(
    title="Mosaic step size (positive float, in arcsec)",
    low=0.01, step=0.01, value=0.20, width=300)

mosaic_step_number_input = Spinner(
    title="Mosaic step number (integer \u2265 1)",
    low=1, step=1, value=5, width=300)

wavelength_select = Select(
    title="Pathloss wavelength (microns)",
    value=wcube_str[0],
    options=wcube_str
)

throughput_switch = Switch(
    label="Showing Throughput", active=True
)

# ---- Dither pattern: count + editable table ---------------------------------
# - dither_pattern_source: FULL table of user-editable X/Y offsets
# - n_dither_input: how MANY of those rows (from the top) are used
dither_pattern_source = ColumnDataSource(
    data=dict(x=list(default_dither_xs),
              y=list(default_dither_ys)))

n_dither_input = Spinner(
    title="Number of dither points to use (integer \u2265 1)",
    low=1, step=1, value=7, width=300)

dither_columns = [
    TableColumn(field='x', title='X offset (arcsec)',
                editor=NumberEditor(step=0.001)),
    TableColumn(field='y', title='Y offset (arcsec)',
                editor=NumberEditor(step=0.001)),
]
dither_table = DataTable(
    source=dither_pattern_source, columns=dither_columns,
    editable=True, width=300, height=400, index_position=None)


def get_active_dither_offsets():
    """Read the dither table and return the (x, y) offsets actually in use.

    "In use" is governed by `n_dither_input`: only the first N rows of
    `dither_pattern_source` (from the top) are returned. This lets the table
    hold more rows (up to 10) than are currently active.

    If `n_dither_input.value` exceeds the number of rows currently in
    `dither_pattern_source`, this pads the table with (0.0, 0.0) rows.

    Returns
    -------
    xs : list of float
        X offsets (arcsec) of the first N dither points.
    ys : list of float
        Y offsets (arcsec) of the first N dither points, same length as
        ``xs``.
    """

    # Get values from widget and table
    n = int(n_dither_input.value)
    xs = list(dither_pattern_source.data['x'])
    ys = list(dither_pattern_source.data['y'])

    # If the table is not long enough, pad with (0, 0) entries
    if n > len(xs):
        pad = n - len(xs)
        xs += [0.0] * pad
        ys += [0.0] * pad
        # Increase size of table and add the (0, 0) entries
        dither_pattern_source.data = dict(x=xs, y=ys)

    # Trim to N first values
    return xs[:n], ys[:n]


# ----- Functions that link widgets to the plot update ------------------------

def update_plot():
    dither_xs, dither_ys = get_active_dither_offsets()
    compute_and_update(
        wcube_str.index(wavelength_select.value),
        throughput_switch.active,
        int(slitlet_shutter_input.value),
        float(mosaic_step_size_input.value),
        int(mosaic_step_number_input.value),
        dither_xs, dither_ys,
        aper_renderer, slitlet_renderer, dith_renderer
    )


def on_widget_change(attr, old, new):
    update_plot()


for widget in (throughput_switch,):
    widget.on_change('active', on_widget_change)
for widget in (wavelength_select,
               slitlet_shutter_input, mosaic_step_size_input,
               mosaic_step_number_input, n_dither_input):
    widget.on_change('value', on_widget_change)

# Table edits (typing a new X/Y offset into a cell) change the source's
# 'data' property, so hook that too.
dither_pattern_source.on_change('data', on_widget_change)


# ---- Initial draw and layout ------------------------------------------------

update_plot()

controls = column(
    row(slitlet_shutter_input, mosaic_step_size_input, mosaic_step_number_input),
    row(throughput_switch, wavelength_select)
)
dither_controls = column(n_dither_input, dither_table)
layout = column(
    Div(text="<b>NIRSpec-IFS shutter/dither/mosaic pattern</b>"),
    row(controls),
    row(p, column(row(dither_controls, p_hist), row(p_xcut, p_ycut)))
)

curdoc().add_root(layout)
curdoc().title = "NIRSpec-IFS Throughput/Coverage Map"