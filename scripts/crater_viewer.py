#!/usr/bin/env python3
"""Interactive crater dataset viewer.

ARCHITECTURE
============
The viewer uses counts.tif as a single master bounding box / reference
coordinate system.  All .map.tif geotiffs are rectangular subregions within
this box, sharing the same lunar stereographic projection and 1 m/pixel scale.

Craters (from craters_v71.csv) are projected once from (lon, lat) into the
reference CRS at startup.  They are rendered as a single ``ax.scatter``
PathCollection — one GPU-friendly draw call for all visible points.
Diameter filtering (1-9 keys) and visibility toggle (c key) rebuild the
scatter.  No per-image crater indexing is needed; craters are always displayed
in their correct reference-frame positions.

A selected .map.tif image is loaded as a separate ``ax.imshow`` layer placed
at its proper extent within the reference frame.  Two levels of detail (LOD)
are used:
  - low-res:  a pre-downsampled image displayed when viewport covers > 25 %
    of the image area.  Fast to pan/zoom.
  - high-res: a cropped region from the full-resolution image, displayed when
    the viewport covers ≤ 25 % of the image area (user has zoomed in).

The LOD check runs on scroll and drag events with a 200 ms debounce.

Navigation
----------
  Mouse wheel over image : zoom centred on cursor
  Left-drag on image     : pan
  Toolbar buttons        : pan (hand), rectangle zoom (magnifier), home (house)

Keyboard
--------
  c     : toggle crater overlay on/off
  1..9  : minimum crater diameter filter (default ≥ 3 m)
  0     : show all craters
  h     : print help to console

UI layout
---------
  Left 20 %  : scrollable file list (CraterListBox) sorted by crater count
  Right 80 % : image panel (ImagePanel) — reference frame + selected image
                 + crater scatter overlay

Data flow
---------
  load_craters()            → (N,3) [lon, lat, diam]
  build_reference()         → CRS + bounds from counts.tif
  project_craters()         → (proj_x, proj_y) in reference CRS
  scan_tiffs() / count_craters_per_file() → file list with crater counts

  CraterViewer.__init__()
    → ImagePanel.__init__()      sets reference viewport
    → ImagePanel.set_crater_data()
        → _rebuild_craters()     creates ax.scatter of all filtered craters
    → on file select:
        → load_image_data()      GDAL ReadAsArray
        → ImagePanel.show_selected_image()
            → downsample
            → ax.imshow(low-res, extent=image_bounds)
            → preload_images()   background thread for neighbours

Known issues (2026-06-16)
==========================
  - LOD switch causes the image to jump horizontally when zooming in.
    The high-res crop extent is computed from xlim/ylim but the crop's
    imshow extent may not match the current viewport, causing a lateral shift.
  - Selected image disappears when viewport moves outside its bounds during
    LOD switch (partially fixed with overlap guard, may still have edge cases).

Usage
=====
    .lunar_venv/bin/python scripts/crater_viewer.py [--tiff-dir DIR] [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# -- GDAL --------------------------------------------------------------------
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from osgeo import gdal, osr

gdal.UseExceptions()

# -- matplotlib --------------------------------------------------------------
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PathCollection, PatchCollection

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULT_TIFF_DIR = "/d/viper/maps/viper_v71_nac"
_CRATER_CSV_V71 = str(Path(__file__).resolve().parent.parent / "data" / "craters_v71.csv")
_CRATER_CSV_ROSS = str(Path(__file__).resolve().parent.parent / "data" / "craters-ross.csv")
_CRATER_CSV = _CRATER_CSV_ROSS  # default at startup
_MOON_RADIUS = 1_737_400.0
_MAX_DISPLAY_DIM = 1600

_EXCLUDE_FILES = {"counts.tif", "resolution.tif"}


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════


def load_craters(csv_path: str) -> np.ndarray:
    """Return (N, 3) float64 array of [lon, lat, diameter_metres].

    Auto-detects column names: lon/lat or center_lon/center_lat.
    """
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        if "center_lon" in fieldnames and "center_lat" in fieldnames:
            lon_key, lat_key = "center_lon", "center_lat"
        elif "lon" in fieldnames and "lat" in fieldnames:
            lon_key, lat_key = "lon", "lat"
        else:
            sys.exit(
                f"Cannot find lon/lat columns in {csv_path}. "
                f"Columns: {fieldnames}"
            )

        for row in reader:
            rows.append(
                (float(row[lon_key]), float(row[lat_key]), float(row["diameter"]))
            )
    if not rows:
        sys.exit(f"No crater rows found in {csv_path}")
    return np.array(rows, dtype=np.float64)


def build_reference(tiff_dir: str):
    """Read counts.tif to establish the master bounding-box coordinate system.

    Returns (crs, x_min, x_max, y_min, y_max) in projected metres.
    """
    counts_path = os.path.join(tiff_dir, "counts.tif")
    if not os.path.isfile(counts_path):
        sys.exit(f"counts.tif not found in {tiff_dir}")

    ds = gdal.Open(counts_path)
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    crs = osr.SpatialReference(ds.GetProjection())
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds = None

    x_min = gt[0]
    x_max = gt[0] + w * gt[1]
    y_min = gt[3] + h * gt[5]
    y_max = gt[3]

    print(
        f"Reference: {w}×{h} m,  x=[{x_min:.0f}, {x_max:.0f}]  y=[{y_min:.0f}, {y_max:.0f}]",
        flush=True,
    )
    return crs, x_min, x_max, y_min, y_max


def project_craters(
    craters: np.ndarray, dst_crs: osr.SpatialReference
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) → (x, y) in the reference CRS."""
    src_crs = osr.SpatialReference()
    src_crs.ImportFromProj4(f"+proj=longlat +R={_MOON_RADIUS} +no_defs")
    src_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    print(f"Projecting {len(craters):,} craters ...", flush=True)
    t0 = time.perf_counter()
    tx = osr.CoordinateTransformation(src_crs, dst_crs)
    proj_x = np.empty(len(craters), dtype=np.float64)
    proj_y = np.empty(len(craters), dtype=np.float64)
    for i in range(len(craters)):
        x, y, _ = tx.TransformPoint(craters[i, 0], craters[i, 1])
        proj_x[i] = x
        proj_y[i] = y
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)
    return proj_x, proj_y


def scan_tiffs(tiff_dir: str) -> list[dict]:
    """Return [{name, path, x_min, x_max, y_min, y_max, crater_count}, ...] sorted by count."""
    files = sorted(
        f
        for f in os.listdir(tiff_dir)
        if f.lower().endswith(".tif") and f not in _EXCLUDE_FILES
    )
    infos = []
    for fn in files:
        path = os.path.join(tiff_dir, fn)
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None
        x_min = gt[0]
        x_max = gt[0] + w * gt[1]
        y_min = gt[3] + h * gt[5]
        y_max = gt[3]
        infos.append(
            {
                "name": fn,
                "path": path,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "crater_count": 0,  # computed below
            }
        )

    # count craters per file
    print(f"Counting craters in {len(infos)} files ...", flush=True)
    t0 = time.perf_counter()
    # We'll count craters after projection, handled by the caller
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)
    return infos


def count_craters_per_file(
    file_infos: list[dict], proj_x: np.ndarray, proj_y: np.ndarray
):
    """Add crater_count to each file_info based on projected coordinates."""
    for fi in file_infos:
        mask = (
            (proj_x >= fi["x_min"])
            & (proj_x <= fi["x_max"])
            & (proj_y >= fi["y_min"])
            & (proj_y <= fi["y_max"])
        )
        fi["crater_count"] = int(mask.sum())

    file_infos.sort(key=lambda fi: fi["crater_count"], reverse=True)


def load_image_data(path: str) -> np.ndarray:
    """Return the full-resolution float32 image array."""
    ds = gdal.Open(path)
    data: np.ndarray = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return data


def compute_contrast(data: np.ndarray) -> tuple[float, float]:
    """Return (vmin, vmax) for sensible display contrast."""
    flat = data.ravel()
    finite = flat[np.isfinite(flat)]
    finite = finite[finite > -1e6]  # filter nodata
    if len(finite) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def downsample(data: np.ndarray, max_dim: int = _MAX_DISPLAY_DIM) -> np.ndarray:
    """Downsample a float32 image to fit within max_dim."""
    h, w = data.shape
    if max(h, w) <= max_dim:
        return data
    scale = max_dim / max(h, w)
    img = Image.fromarray(data)
    return np.array(
        img.resize((int(w * scale), int(h * scale)), Image.BILINEAR), dtype=np.float32
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scrollable file list
# ═════════════════════════════════════════════════════════════════════════════


class CraterListBox:
    ZEBRA_EVEN = "#252525"
    ZEBRA_ODD = "#1e1e1e"
    TEXT_COLOR = "#cccccc"
    COUNT_COLOR = "#888888"
    SEL_BG = "#3a6ea5"
    SEL_TEXT = "#ffffff"
    SCROLLBAR_COLOR = "#555555"
    SCROLLBAR_BG = "#333333"
    FONT_SIZE = 10

    def __init__(self, ax: plt.Axes, file_infos: list[dict], on_select=None):
        self.ax = ax
        self.file_infos = file_infos
        self.on_select = on_select
        self._scroll_offset = 0
        self._selected_idx: int | None = None
        self._visible_items = 30
        self._ever_drawn = False
        self._last_scroll_time = 0.0
        self._setup_axes()
        self._connect_events()

    def _compute_visible_items(self) -> int:
        bbox = self.ax.get_window_extent()
        if bbox.height <= 0:
            return 30
        return max(4, int(bbox.height / (self.FONT_SIZE * 2.6)))

    def _setup_axes(self):
        self.ax.set_facecolor(self.ZEBRA_ODD)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self._text_x = 0.02
        self._text_x_end = 0.88
        self._scroll_x = 0.91
        self._scroll_w = 0.07

    def _connect_events(self):
        canvas = self.ax.figure.canvas
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("resize_event", self._on_resize)
        canvas.mpl_connect("draw_event", self._on_first_draw)

    def _on_first_draw(self, event):
        if not self._ever_drawn:
            self._ever_drawn = True
            self._redraw()

    def _redraw(self):
        ax = self.ax
        for artist in list(ax.patches) + list(ax.texts) + list(ax.lines):
            artist.remove()

        self._visible_items = self._compute_visible_items()
        n_visible = self._visible_items
        n = len(self.file_infos)
        ax.set_xlim(0, 1)
        ax.set_ylim(n_visible + 0.5, -0.5)

        # header
        header_y = n_visible + 0.3
        ax.axhline(y=n_visible - 0.1, color="#555555", linewidth=1, xmin=0, xmax=0.88)
        ax.text(
            self._text_x, header_y, "file",
            fontsize=self.FONT_SIZE - 1, fontfamily="monospace",
            color="#777777", va="bottom", ha="left", zorder=3,
        )
        ax.text(
            self._text_x_end, header_y, "#",
            fontsize=self.FONT_SIZE - 1, fontfamily="monospace",
            color="#777777", va="bottom", ha="right", zorder=3,
        )

        visible_start = max(0, self._scroll_offset)
        visible_end = min(n, visible_start + n_visible)

        for vi, i in enumerate(range(visible_start, visible_end)):
            y_bottom = n_visible - vi - 1
            y_center = y_bottom + 0.5
            is_sel = i == self._selected_idx
            display_name = Path(self.file_infos[i]["name"]).stem.replace(".map", "")
            count = self.file_infos[i]["crater_count"]

            bg_color = (
                self.SEL_BG
                if is_sel
                else (self.ZEBRA_EVEN if i % 2 == 0 else self.ZEBRA_ODD)
            )
            ax.add_patch(
                Rectangle((0, y_bottom), 1, 1, facecolor=bg_color, edgecolor="none", zorder=0)
            )
            ax.text(
                self._text_x, y_center, f" {display_name}",
                fontsize=self.FONT_SIZE, fontfamily="monospace",
                color=self.SEL_TEXT if is_sel else self.TEXT_COLOR,
                va="center", ha="left", zorder=2,
            )
            ax.text(
                self._text_x_end, y_center, str(count),
                fontsize=self.FONT_SIZE - 1, fontfamily="monospace",
                color=self.SEL_TEXT if is_sel else self.COUNT_COLOR,
                va="center", ha="right", zorder=2,
            )

        if n > n_visible and n_visible > 0:
            max_scroll = n - n_visible
            thumb_h = max(0.03, n_visible / n)
            thumb_ratio = self._scroll_offset / max_scroll
            thumb_bottom = (1.0 - thumb_h) * (1.0 - thumb_ratio)
            ax.add_patch(
                Rectangle(
                    (self._scroll_x, 0), self._scroll_w, 1,
                    transform=ax.transAxes, facecolor=self.SCROLLBAR_BG,
                    edgecolor="none", zorder=1,
                )
            )
            ax.add_patch(
                Rectangle(
                    (self._scroll_x, thumb_bottom), self._scroll_w, thumb_h,
                    transform=ax.transAxes, facecolor=self.SCROLLBAR_COLOR,
                    edgecolor="none", zorder=2,
                )
            )

        ax.figure.canvas.draw_idle()

    def _on_resize(self, event):
        if self._ever_drawn:
            self._redraw()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        if not self._ever_drawn:
            return
        self._last_scroll_time = time.perf_counter()
        delta = -1 if event.button == "up" else 1
        n = len(self.file_infos)
        self._scroll_offset = max(
            0, min(max(0, n - self._visible_items), self._scroll_offset + delta)
        )
        self._redraw()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.button != MouseButton.LEFT:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        # scrollbar click → page up/down
        if x >= self._scroll_x:
            n = len(self.file_infos)
            thumb_h = max(0.03, self._visible_items / n) if n > 0 else 0.03
            track_click_y = y / self._visible_items
            max_scroll = max(0, n - self._visible_items)
            if n > self._visible_items:
                thumb_ratio = self._scroll_offset / max(1, max_scroll)
                thumb_bottom = (1.0 - thumb_h) * (1.0 - thumb_ratio)
                thumb_top = thumb_bottom + thumb_h
                if track_click_y < thumb_bottom:
                    self._scroll_offset = max(0, self._scroll_offset - self._visible_items)
                elif track_click_y > thumb_top:
                    self._scroll_offset = min(max_scroll, self._scroll_offset + self._visible_items)
                self._redraw()
            return
        # debounce trackpad scroll-noise clicks
        if time.perf_counter() - self._last_scroll_time < 0.200:
            return
        row = int(self._visible_items - y)
        idx = self._scroll_offset + row
        if 0 <= idx < len(self.file_infos):
            self._selected_idx = idx
            self._redraw()
            if self.on_select is not None:
                self.on_select(idx)


# ═════════════════════════════════════════════════════════════════════════════
# Image panel — unified reference coordinate system
# ═════════════════════════════════════════════════════════════════════════════


class ImagePanel:
    """Right-side panel: selected image + crater overlay in reference coordinates.

    Craters are rendered as a single ax.scatter PathCollection — one GPU-friendly
    draw call for all points.  The artist is stored and updated in-place on filter
    changes to avoid re-creation overhead.
    """

    def __init__(self, ax: plt.Axes, ref_x_min, ref_x_max, ref_y_min, ref_y_max):
        self.ax = ax
        self.ref_x_min = ref_x_min
        self.ref_x_max = ref_x_max
        self.ref_y_min = ref_y_min
        self.ref_y_max = ref_y_max

        self.ax.set_facecolor("#0d0d0d")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # crater data (global reference frame)
        self._crater_xs: np.ndarray | None = None
        self._crater_ys: np.ndarray | None = None
        self._crater_diams: np.ndarray | None = None
        self._show_craters: bool = True
        self._min_diameter: float = 3.0
        self._crater_scatter: PathCollection | PatchCollection | None = None
        self._crater_mode: str = "dots"  # "dots" or "circles"
        self._CRATER_CIRCLE_THRESHOLD = 2000  # switch to circles when ≤ this many visible

        # LOD: selected-image data
        self._full_image: np.ndarray | None = None
        self._full_image_extent: tuple | None = None
        self._low_res_image: np.ndarray | None = None
        self._imshow_obj = None
        self._current_lod: str = "low"
        self._current_image_path: str = ""
        self._last_lod_check: float = 0.0

        # contrast cache per path
        self._contrast_cache: dict[str, tuple[float, float]] = {}

        # background pre-load
        self._preload_lock = threading.Lock()
        self._preloaded: dict[str, np.ndarray] = {}
        self._preload_pending: set[str] = set()

        # initial viewport: full reference extent
        self.ax.set_xlim(ref_x_min, ref_x_max)
        self.ax.set_ylim(ref_y_min, ref_y_max)  # bottom=south, top=north
        self.ax.set_title("Select a file from the list", fontsize=10, color="#aaaaaa")
        self.ax.set_aspect("equal")

    # -- crater data -----------------------------------------------------------

    def set_crater_data(self, xs, ys, diams):
        self._crater_xs = xs
        self._crater_ys = ys
        self._crater_diams = diams
        self._rebuild_craters()

    def toggle_craters(self) -> bool:
        self._show_craters = not self._show_craters
        self._rebuild_craters()
        return self._show_craters

    def set_min_diameter(self, min_d: float):
        self._min_diameter = min_d
        self._rebuild_craters()

    # -- crater scatter --------------------------------------------------------

    def _rebuild_craters(self):
        """(Re)create the scatter artist with craters matching the current filter."""
        # remove old
        if self._crater_scatter is not None:
            self._crater_scatter.remove()
            self._crater_scatter = None

        if not self._show_craters:
            self.ax.figure.canvas.draw_idle()
            return
        if self._crater_xs is None or len(self._crater_xs) == 0:
            return

        # filter by minimum diameter
        if self._min_diameter > 0:
            keep = self._crater_diams >= self._min_diameter
            if not keep.any():
                self.ax.figure.canvas.draw_idle()
                return
            xs = self._crater_xs[keep]
            ys = self._crater_ys[keep]
        else:
            xs = self._crater_xs
            ys = self._crater_ys

        # Draw all filtered craters as a single scatter — matplotlib handles
        # viewport clipping automatically.  Using small marker size for speed.
        self._crater_scatter = self.ax.scatter(
            xs,
            ys,
            s=2.0,
            c="#ff2222",
            marker="o",
            linewidths=0,
            alpha=0.6,
            zorder=5,
            rasterized=True,
        )
        self._crater_mode = "dots"
        self.ax.figure.canvas.draw_idle()

    def _crater_lod_check(self):
        """Switch between scatter dots (many craters) and circles (few craters)."""
        if not self._show_craters:
            return
        if self._crater_xs is None or len(self._crater_xs) == 0:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        vx1, vx2 = sorted(xlim)
        vy1, vy2 = sorted(ylim)

        # count craters in viewport
        in_view = (
            (self._crater_xs >= vx1)
            & (self._crater_xs <= vx2)
            & (self._crater_ys >= vy1)
            & (self._crater_ys <= vy2)
        )
        if self._min_diameter > 0:
            in_view &= self._crater_diams >= self._min_diameter

        n_visible = int(in_view.sum())

        if n_visible <= self._CRATER_CIRCLE_THRESHOLD and self._crater_mode != "circles":
            # switch to circle mode — data-coordinate patches showing true diameter
            if self._crater_scatter is not None:
                self._crater_scatter.remove()

            idx = np.where(in_view)[0]
            patches = [
                Circle(
                    (self._crater_xs[i], self._crater_ys[i]),
                    radius=self._crater_diams[i] * 0.5,
                )
                for i in idx
            ]
            self._crater_scatter = PatchCollection(
                patches,
                facecolor="none",
                edgecolor="#ff4444",
                linewidths=0.5,
                alpha=0.7,
                zorder=5,
                match_original=True,
            )
            self.ax.add_collection(self._crater_scatter)
            self._crater_mode = "circles"
            self.ax.figure.canvas.draw_idle()

        elif n_visible > self._CRATER_CIRCLE_THRESHOLD and self._crater_mode != "dots":
            # switch back to dot mode — rebuild all craters as scatter
            self._rebuild_craters()

    # -- image loading ---------------------------------------------------------

    def show_counts_background(self, path: str):
        """Load and display counts.tif as the initial background."""
        full_res = load_image_data(path)
        vmin, vmax = compute_contrast(full_res)
        low_res = downsample(full_res)
        display = np.clip((low_res - vmin) / (vmax - vmin), 0.0, 1.0)

        if self._imshow_obj is not None:
            self._imshow_obj.remove()

        self._imshow_obj = self.ax.imshow(
            display,
            cmap="gray",
            extent=[self.ref_x_min, self.ref_x_max, self.ref_y_min, self.ref_y_max],
            aspect="equal",
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )
        self.ax.set_title("counts  —  reference frame", fontsize=10, color="#aaaaaa")
        self.ax.figure.canvas.draw_idle()

    def show_selected_image(self, full_res: np.ndarray, path: str, x1, x2, y1, y2, title: str):
        """Place a geotiff image at its proper extent within the reference frame."""
        if path not in self._contrast_cache:
            self._contrast_cache[path] = compute_contrast(full_res)
        vmin, vmax = self._contrast_cache[path]

        self._full_image = full_res
        self._full_image_extent = (x1, x2, y1, y2)
        self._current_lod = "low"
        self._current_image_path = path

        self._low_res_image = downsample(full_res)
        display = np.clip((self._low_res_image - vmin) / (vmax - vmin), 0.0, 1.0)

        if self._imshow_obj is not None:
            self._imshow_obj.remove()

        self._imshow_obj = self.ax.imshow(
            display,
            cmap="gray",
            extent=[x1, x2, y1, y2],
            aspect="equal",
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )
        self.ax.set_title(title, fontsize=10, color="#aaaaaa")
        self.ax.figure.canvas.draw_idle()

    def preload_images(self, file_infos: list[dict], around_idx: int):
        indices = list(
            range(max(0, around_idx - 2), min(len(file_infos), around_idx + 3))
        )
        for i in indices:
            if i == around_idx:
                continue
            fi = file_infos[i]
            if fi["path"] in self._preloaded or fi["path"] in self._preload_pending:
                continue
            self._preload_pending.add(fi["path"])
            t = threading.Thread(
                target=self._preload_worker, args=(fi["path"],), daemon=True
            )
            t.start()

    def _preload_worker(self, path: str):
        try:
            data = load_image_data(path)
            with self._preload_lock:
                self._preloaded[path] = data
                self._preload_pending.discard(path)
        except Exception:
            self._preload_pending.discard(path)

    def get_preloaded(self, path: str) -> np.ndarray | None:
        with self._preload_lock:
            return self._preloaded.pop(path, None)

    # -- LOD -------------------------------------------------------------------

    def check_lod(self):
        now = time.perf_counter()
        if now - self._last_lod_check < 0.200:
            return
        self._last_lod_check = now
        self._lod_refresh()
        self._crater_lod_check()

    def _lod_refresh(self):
        if self._full_image is None or self._imshow_obj is None or self._full_image_extent is None:
            return
        if self._low_res_image is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x1, x2, y1, y2 = self._full_image_extent
        img_w = x2 - x1
        img_h = y2 - y1
        if img_w == 0 or img_h == 0:
            return

        view_w = xlim[1] - xlim[0]
        view_h = abs(ylim[1] - ylim[0])
        coverage = (view_w * view_h) / (img_w * img_h)

        path = self._current_image_path
        if path in self._contrast_cache:
            vmin, vmax = self._contrast_cache[path]
        else:
            vmin, vmax = compute_contrast(self._full_image)
            self._contrast_cache[path] = (vmin, vmax)

        if coverage > 0.25:
            # zoomed out enough — use low-res
            if self._current_lod != "low":
                display = np.clip((self._low_res_image - vmin) / (vmax - vmin), 0.0, 1.0)
                self._imshow_obj.set_data(display)
                self._imshow_obj.set_extent([x1, x2, y1, y2])
                self._current_lod = "low"
                self.ax.figure.canvas.draw_idle()
            return

        # coverage <= 0.25: zoomed in — crop high-res
        if xlim[1] < x1 or xlim[0] > x2:
            return  # entirely left/right of image
        if ylim[1] < min(y1, y2) or ylim[0] > max(y1, y2):
            return  # entirely above/below image

        margin = 0.3
        vx1, vx2 = sorted(xlim)
        vy1, vy2 = sorted(ylim)
        pad_w = (vx2 - vx1) * margin
        pad_h = (vy2 - vy1) * margin
        vx1, vx2 = max(x1, vx1 - pad_w), min(x2, vx2 + pad_w)
        vy1, vy2 = max(min(y1, y2), vy1 - pad_h), min(max(y1, y2), vy2 + pad_h)

        px1 = max(0, int(vx1 - x1))
        px2 = min(self._full_image.shape[1], int(np.ceil(vx2 - x1)))
        # origin="upper": row 0 = y2 (north), row H-1 = y1 (south)
        # pixel row for a given y:  r = y2 - y
        py_top = max(0, int(y2 - vy2))     # first row of crop (north edge)
        py_bot = min(self._full_image.shape[0], int(np.ceil(y2 - vy1)))  # last row (south edge)
        py1, py2 = py_top, py_bot

        if px1 >= px2 or py1 >= py2:
            return

        crop = self._full_image[py1:py2, px1:px2]
        if crop.size == 0:
            return
        crop = downsample(crop)

        display = np.clip((crop - vmin) / (vmax - vmin), 0.0, 1.0)
        self._imshow_obj.set_data(display)
        self._imshow_obj.set_extent([vx1, vx2, vy1, vy2])
        self._current_lod = "high"
        self.ax.figure.canvas.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
# Main viewer
# ═════════════════════════════════════════════════════════════════════════════


class CraterViewer:
    def __init__(
        self,
        file_infos: list[dict],
        proj_x: np.ndarray,
        proj_y: np.ndarray,
        crater_diams: np.ndarray,
        ref_x_min, ref_x_max, ref_y_min, ref_y_max,
        tiff_dir: str,
        ref_crs,
        csv_paths: list[str],
        current_csv_idx: int,
    ):
        self.file_infos = file_infos
        self._ref_crs = ref_crs
        self._csv_paths = csv_paths
        self._current_csv_idx = current_csv_idx
        self._tiff_dir = tiff_dir

        self.fig = plt.figure(
            "Crater Viewer — c:toggle craters  1-9:min-diameter  0:all  h:help",
            figsize=(18, 10), facecolor="#1a1a1a",
        )
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.002)

        self.list_ax = self.fig.add_subplot(gs[0, 0])
        self.img_ax = self.fig.add_subplot(gs[0, 1])

        self.listbox = CraterListBox(self.list_ax, file_infos, on_select=self._on_file_select)
        self.img_panel = ImagePanel(self.img_ax, ref_x_min, ref_x_max, ref_y_min, ref_y_max)
        self.img_panel.set_crater_data(proj_x, proj_y, crater_diams)

        # Show counts.tif as the initial background
        counts_path = os.path.join(tiff_dir, "counts.tif")
        self.img_panel.show_counts_background(counts_path)

        # Drag-to-pan state
        self._pan_start_x: float | None = None
        self._pan_start_y: float | None = None
        self._pan_xlim_start = None
        self._pan_ylim_start = None

        self.fig.canvas.mpl_connect("scroll_event", self._on_img_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_img_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_img_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_img_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        self._print_help()
        self._update_window_title()
        plt.show()

    def _print_help(self):
        print(
            "\nKeyboard shortcuts:\n"
            "  c          Toggle crater overlay on/off\n"
            "  1..9       Minimum crater diameter filter (default: 3m)\n"
            "  0          Show all craters\n"
            "  h          Print this help\n"
            "  Mouse wheel over image: zoom centred on cursor\n"
            "  Left-drag on image: pan\n"
            "  Toolbar: pan (hand), rectangle zoom (magnifier), home (house)\n",
            flush=True,
        )

    def _on_resize(self, event):
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    def _on_file_select(self, idx: int):
        fi = self.file_infos[idx]
        print(f"Loading {fi['name']} ...", flush=True)
        t0 = time.perf_counter()

        full_res = load_image_data(fi["path"])
        elapsed = time.perf_counter() - t0

        title = (
            f"{Path(fi['name']).stem.replace('.map', '')}"
            f"  —  {fi['crater_count']} craters  —  {elapsed:.1f}s"
            f"{'  [overlay OFF]' if not self.img_panel._show_craters else ''}"
        )
        print(f"  {full_res.shape} image, {fi['crater_count']} craters in {elapsed:.1f}s", flush=True)

        self.img_panel.show_selected_image(
            full_res, fi["path"], fi["x_min"], fi["x_max"], fi["y_min"], fi["y_max"], title
        )
        self.img_panel.preload_images(self.file_infos, idx)

    # -- mouse-wheel zoom ---------------------------------------------------

    def _on_img_scroll(self, event):
        if event.inaxes is not self.img_ax:
            return
        inv = self.img_ax.transData.inverted()
        x_center, y_center = inv.transform((event.x, event.y))
        xlim = self.img_ax.get_xlim()
        ylim = self.img_ax.get_ylim()
        scale_factor = 1.15 if event.button == "up" else 1 / 1.15
        new_x_range = (xlim[1] - xlim[0]) / scale_factor
        new_y_range = (ylim[1] - ylim[0]) / scale_factor
        self.img_ax.set_xlim(
            x_center - new_x_range * (x_center - xlim[0]) / (xlim[1] - xlim[0]),
            x_center + new_x_range * (xlim[1] - x_center) / (xlim[1] - xlim[0]),
        )
        self.img_ax.set_ylim(
            y_center - new_y_range * (y_center - ylim[0]) / (ylim[1] - ylim[0]),
            y_center + new_y_range * (ylim[1] - y_center) / (ylim[1] - ylim[0]),
        )
        self.img_panel.check_lod()
        self.img_ax.figure.canvas.draw_idle()

    # -- drag-to-pan ---------------------------------------------------------

    def _on_img_press(self, event):
        if event.inaxes is not self.img_ax:
            return
        if event.button != MouseButton.LEFT:
            return
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._pan_xlim_start = self.img_ax.get_xlim()
        self._pan_ylim_start = self.img_ax.get_ylim()

    def _on_img_release(self, event):
        self._pan_start_x = None
        self._pan_start_y = None

    def _on_img_motion(self, event):
        if (
            self._pan_start_x is None
            or self._pan_xlim_start is None
            or self._pan_ylim_start is None
        ):
            return
        if event.inaxes is not self.img_ax:
            return
        dx_px = self._pan_start_x - event.x
        dy_px = self._pan_start_y - event.y
        if dx_px is None or dy_px is None:
            return
        bbox = self.img_ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        xlim0 = self._pan_xlim_start
        ylim0 = self._pan_ylim_start
        dx_data = dx_px * (xlim0[1] - xlim0[0]) / bbox.width
        dy_data = dy_px * (ylim0[1] - ylim0[0]) / bbox.height
        self.img_ax.set_xlim(xlim0[0] + dx_data, xlim0[1] + dx_data)
        self.img_ax.set_ylim(ylim0[0] + dy_data, ylim0[1] + dy_data)
        self.img_panel.check_lod()
        self.img_ax.figure.canvas.draw_idle()

    # -- keyboard -------------------------------------------------------------

    def _on_keypress(self, event):
        if event.key == "c":
            state = self.img_panel.toggle_craters()
            print(f"Crater overlay: {'ON' if state else 'OFF'}", flush=True)
            self._update_title()
        elif event.key in "123456789":
            min_d = int(event.key)
            self.img_panel._show_craters = True
            self.img_panel.set_min_diameter(min_d)
            print(f"Crater min diameter: {min_d}m", flush=True)
            self._update_title()
        elif event.key == "0":
            self.img_panel._show_craters = True
            self.img_panel.set_min_diameter(0.0)
            print("Showing all craters", flush=True)
            self._update_title()
        elif event.key == "v":
            self._toggle_csv()
        elif event.key == "h":
            self._print_help()

    def _update_title(self):
        if self.listbox._selected_idx is not None:
            fi = self.file_infos[self.listbox._selected_idx]
            show = self.img_panel._show_craters
            min_d = self.img_panel._min_diameter
            parts = [
                Path(fi["name"]).stem.replace(".map", ""),
                f"{fi['crater_count']} craters",
            ]
            if not show:
                parts.append("[overlay OFF]")
            elif min_d > 0:
                parts.append(f"[>={min_d:.0f}m]")
            self.img_panel.ax.set_title("  —  ".join(parts), fontsize=10, color="#aaaaaa")
            self.fig.canvas.draw_idle()

    def _csv_name(self) -> str:
        return Path(self._csv_paths[self._current_csv_idx]).stem

    def _update_window_title(self):
        name = self._csv_name()
        self.fig.canvas.manager.set_window_title(
            f"Crater Viewer — {name}  c:toggle  v:switch-csv  1-9:filter  0:all  h:help"
        )

    def _toggle_csv(self):
        """Switch to the other CSV, reload & re-project craters."""
        self._current_csv_idx = (self._current_csv_idx + 1) % len(self._csv_paths)
        csv_path = self._csv_paths[self._current_csv_idx]
        name = self._csv_name()

        print(f"Switching to {name} ...", flush=True)
        t0 = time.perf_counter()
        craters = load_craters(csv_path)
        proj_x, proj_y = project_craters(craters, self._ref_crs)
        count_craters_per_file(self.file_infos, proj_x, proj_y)
        elapsed = time.perf_counter() - t0

        total = sum(fi["crater_count"] for fi in self.file_infos)
        print(f"  {len(craters):,} craters in {elapsed:.1f}s, {total:,} total hits", flush=True)

        # Update crater data and redraw
        self.img_panel.set_crater_data(proj_x, proj_y, craters[:, 2])

        # Re-sort and redraw file list
        self.file_infos.sort(key=lambda fi: fi["crater_count"], reverse=True)
        self.listbox._selected_idx = None
        self.listbox._scroll_offset = 0
        self.listbox.file_infos = self.file_infos
        self.listbox._redraw()

        self._update_window_title()
        self._update_title()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Crater dataset viewer")
    p.add_argument("--tiff-dir", default=_DEFAULT_TIFF_DIR)
    p.add_argument("--csv", default=_CRATER_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.tiff_dir):
        sys.exit(f"TIFF directory not found: {args.tiff_dir}")
    if not os.path.isfile(args.csv):
        sys.exit(f"CSV file not found: {args.csv}")

    print(f"Loading craters from {args.csv} ...", flush=True)
    t0 = time.perf_counter()
    craters = load_craters(args.csv)
    print(f"  {len(craters):,} craters in {time.perf_counter() - t0:.1f}s")

    # Reference frame from counts.tif
    ref_crs, rx1, rx2, ry1, ry2 = build_reference(args.tiff_dir)

    # Project once
    proj_x, proj_y = project_craters(craters, ref_crs)

    # Scan map.tif files
    file_infos = scan_tiffs(args.tiff_dir)
    count_craters_per_file(file_infos, proj_x, proj_y)

    total = sum(fi["crater_count"] for fi in file_infos)
    print(f"Indexed {len(file_infos)} files, {total:,} total crater hits", flush=True)
    print("Launching viewer ...", flush=True)

    csv_paths = [_CRATER_CSV_V71, _CRATER_CSV_ROSS]
    current_idx = 0 if args.csv == _CRATER_CSV_V71 else 1
    CraterViewer(file_infos, proj_x, proj_y, craters[:, 2], rx1, rx2, ry1, ry2, args.tiff_dir, ref_crs, csv_paths, current_idx)


if __name__ == "__main__":
    main()
