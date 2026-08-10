# Use bokeh to allow user interactivity
# All units in arcseconds
# X is dispersion direction (along short shutter length)
# Y is cross-dispersion direction (along slitlet)

# ---- Imports ----------------------------------------------------------------

from bokeh.plotting import figure, curdoc
from bokeh.models import (Arrow, VeeHead, ColumnDataSource, Spinner, Div,
                          DataTable, TableColumn, NumberEditor)
from bokeh.layouts import column, row


# ---- Fixed shutter geometry -------------------------------------------------

shutter_width = 0.2
shutter_height = 0.46
shutter_gap = 0.07
shutter_x = 0
shutter_y = 0


# ---- Default dither pattern -------------------------------------------------

# "5-point dither" mimicking what DLaw showed in the meeting
default_dither_xs = [0, 0.165, 0.165, 0, 0.0825, 0, 0, 0, 0, 0]
default_dither_ys = [0, 0, 0.425, 0.425, 0.2125, 0, 0, 0, 0, 0]


# ---- Figure and data sources ------------------------------------------------
p = figure(width=1000, height=800, match_aspect=True)
p.xaxis.axis_label = "Arcsecond - dispersed direction"
p.yaxis.axis_label = "Arcsecond - cross-dispersed direction"

mosaic_source = ColumnDataSource(data=dict(x=[], y=[]))
dither_source = ColumnDataSource(data=dict(x=[], y=[]))
slitlet_source = ColumnDataSource(data=dict(x=[], y=[]))

# Show all shutters coverage
p.rect(source=mosaic_source, x='x', y='y',
       width=shutter_width, height=shutter_height,
       color="#CAB2D6", alpha=0.25, line_color='black')
# Show dither pattern shutters centers
p.scatter(source=dither_source, x='x', y='y',
          marker='cross', color='black', size=10)
# Show origin slitlets
p.rect(source=slitlet_source, x='x', y='y',
       width=shutter_width, height=shutter_height,
       fill_color=None, line_color='black')
# Track mosaic step arrows so we can remove/replace them
vh = VeeHead(size=15, fill_color='red')
arrow_renderers = []


def compute_and_update(slitlet_shutter, mosaic_step_size, mosaic_step_number,
                       dither_xs, dither_ys):
    """
    Recomputes all geometry and push it into the plot.
    """

    # Slitlet's shutters positions
    slitlet_xs = [shutter_x for _ in range(slitlet_shutter)]
    slitlet_ys = [shutter_y + i * (shutter_height + shutter_gap)
                  for i in range(slitlet_shutter)]

    # All dither's shutters positions
    all_dither_xs = [ssx + dx for ssx in slitlet_xs for dx in dither_xs]
    all_dither_ys = [ssy + dy for ssy in slitlet_ys for dy in dither_ys]

    # Dither pattern shutter centers (for the origin slitlet)
    ori_dither_xs = [shutter_x + dx for dx in dither_xs]
    ori_dither_ys = [shutter_y + dy for dy in dither_ys]

    # All mosaic's shutters positions
    all_mosaic_xs = [i * mosaic_step_size + dx
                     for i in range(mosaic_step_number) for dx in
                     all_dither_xs]
    all_mosaic_ys = [dy for i in range(mosaic_step_number) for dy in
                     all_dither_ys]

    # Mosaic's first shutter positions (for the arrows)
    ori_mosaic_xs = [i * mosaic_step_size + shutter_x for i in
                     range(mosaic_step_number)]
    ori_mosaic_ys = [shutter_y for i in range(mosaic_step_number)]


    # Push new data into the sources (triggers redraw)
    mosaic_source.data = dict(x=all_mosaic_xs, y=all_mosaic_ys)
    dither_source.data = dict(x=ori_dither_xs, y=ori_dither_ys)
    slitlet_source.data = dict(x=slitlet_xs, y=slitlet_ys)

    # Rebuild the mosaic arrows (their count depends on mosaic_step_number).
    if arrow_renderers:
        p.center = [c for c in p.center if c not in arrow_renderers]
    arrow_renderers.clear()
    for i in range(mosaic_step_number - 1):
        arrow = Arrow(x_start=ori_mosaic_xs[i], y_start=ori_mosaic_ys[i],
                      x_end=ori_mosaic_xs[i + 1], y_end=ori_mosaic_ys[i + 1],
                      end=vh, line_color='red')
        p.add_layout(arrow)
        arrow_renderers.append(arrow)


# ---- Input widgets ----------------------------------------------------------
slitlet_shutter_input = Spinner(
    title="Shutters per slitlet (integer \u2265 1)",
    low=1, step=1, value=3, width=300)

mosaic_step_size_input = Spinner(
    title="Mosaic step size (positive float, in arcsec)",
    low=0.01, step=0.01, value=.15, width=300)

mosaic_step_number_input = Spinner(
    title="Mosaic step number (integer \u2265 1)",
    low=1, step=1, value=5, width=300)


# ---- Dither pattern: count + editable table ---------------------------------
# - dither_pattern_source: FULL table of user-editable X/Y offsets
# - n_dither_input: how MANY of those rows (from the top) are used
dither_pattern_source = ColumnDataSource(
    data=dict(x=list(default_dither_xs), y=list(default_dither_ys)))

n_dither_input = Spinner(
    title="Number of dither points to use (integer \u2265 1)",
    low=1, step=1, value=5, width=220)

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
    """Return the (x, y) offset lists actually used, based on the count spinner.

    If the count exceeds the number of rows currently in the table, the
    table is padded with (0, 0) rows up to that count. If it's smaller,
    only the first N rows are used (extra rows stay in the table, unused).
    """
    n = int(n_dither_input.value)
    xs = list(dither_pattern_source.data['x'])
    ys = list(dither_pattern_source.data['y'])
    if n > len(xs):
        pad = n - len(xs)
        xs += [0.0] * pad
        ys += [0.0] * pad
        # Reflect the padding back into the table so the user sees the new
        # empty rows ready to be filled in.
        dither_pattern_source.data = dict(x=xs, y=ys)
    return xs[:n], ys[:n]


def update_plot():
    dither_xs, dither_ys = get_active_dither_offsets()
    compute_and_update(
        int(slitlet_shutter_input.value),
        float(mosaic_step_size_input.value),
        int(mosaic_step_number_input.value),
        dither_xs, dither_ys,
    )


def on_widget_change(attr, old, new):
    update_plot()


for widget in (slitlet_shutter_input, mosaic_step_size_input,
               mosaic_step_number_input, n_dither_input):
    widget.on_change('value', on_widget_change)

# Table edits (typing a new X/Y offset into a cell) change the source's
# 'data' property, so hook that too.
dither_pattern_source.on_change('data', on_widget_change)


# ---- Initial draw and layout ------------------------------------------------

update_plot()

controls = row(slitlet_shutter_input, mosaic_step_size_input,
               mosaic_step_number_input)
dither_controls = column(n_dither_input, dither_table)
layout = column(
    Div(text="<b>NIRSpec-IFS shutter/dither/mosaic pattern</b>"),
    row(controls),
    row(p, dither_controls),
)

curdoc().add_root(layout)
curdoc().title = "NIRSpec-IFS coverage"
