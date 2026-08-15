"""Pure-numpy selection core for the TRGB pipeline.

Per-star array construction, the spatial/color/magnitude selection masks, and
the fast continuous-edge tip estimate. No plotting dependency, so it runs
headless and is cheap enough to re-evaluate repeatedly.
"""

import numpy as np

import photometry
from edge_detection import edge_detect
# Edge detector window range: single source of truth in galaxy_configs,
# re-exported here so edge_tip's defaults stay in sync with the pipeline's.
from galaxy_configs import DEFAULT_COLOR, TIP_LO, TIP_HI
# How many evenly spaced points for the edge detector curve based on `TIP_LO` and `TIP_HI`
N_GRID = 500


def build_arrays(df, color_correct=True):
    """Per-star arrays (as opposed to binned-histograms) the selection filters cut on, computed once.

    The pipeline's magnitude construction: color-rectified F814W (photometry
    .color_correct) and the raw catalog e_F814W. color_correct=False keeps
    the raw F814W magnitude instead (pipeline parity is the default)."""
    color = (df["F606W_0"] - df["F814W_0"]).to_numpy()
    if color_correct:
        mag = photometry.color_correct(df["F814W_0"],
                                       df["F606W_0"] - df["F814W_0"]).to_numpy()
    else:
        mag = df["F814W_0"].to_numpy()
    return {
        "ra": df["ra"].to_numpy(),
        "dec": df["dec"].to_numpy(),
        "color": color,
        "mag": mag,
        "err": df["e_F814W"].to_numpy(),
    }


def ellipse_mask(ra, dec, ra_cen, dec_cen, a, b, pa=0.0):
    """Tangent-plane ellipse containment a/b in arcsec, pa in deg E of N."""
    east = (ra - ra_cen) * np.cos(np.radians(dec_cen)) * 3600.0
    north = (dec - dec_cen) * 3600.0
    sin_pa, cos_pa = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    u = east * sin_pa + north * cos_pa    # along the major axis (arcsec)
    v = east * cos_pa - north * sin_pa    # along the minor axis
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def ellipse_outline(ra_cen, dec_cen, a, b, pa=0.0, n=181):
    """(ra, dec) polyline of the selection ellipse; the inverse of the
    ellipse_mask decomposition, so the drawn aperture is always the applied one. The polyline is determined from the ellipse_mask."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    u, v = a * np.cos(t), b * np.sin(t)
    sin_pa, cos_pa = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    east = u * sin_pa + v * cos_pa
    north = u * cos_pa - v * sin_pa
    ra = ra_cen + east / (3600.0 * np.cos(np.radians(dec_cen)))
    dec = dec_cen + north / 3600.0
    return ra, dec


def polygon_mask(ra, dec, verts):
    """Boolean mask of which (ra, dec) points fall inside the freehand
    polygon ``verts``, via an even-odd ray-casting test vectorized over the
    points. This is the pencil-tool counterpart of ellipse_mask."""
    v = np.asarray(verts, dtype=float)
    x = np.asarray(ra, dtype=float)
    y = np.asarray(dec, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    x0, y0 = v[-1]
    for x1, y1 in v:
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < xc)
        x0, y0 = x1, y1
    return inside


def apply_selection(cat, sel):
    """Boolean masks for a filter set over the star arrays.

    ``sel`` keys: ra_cen, dec_cen, a, b, pa (arcsec/deg E of N), mode
    ("inside"/"outside"/"off"), color_min, color_max, mag_bright, mag_faint.
    Optional pencil keys: spatial_tool ("ellipse", the default, or
    "pencil") with pencil_verts, a freehand (ra, dec) polygon that replaces
    the ellipse as the outer spatial region."""

    verts = sel.get("pencil_verts")
    sub_verts = sel.get("pencil_sub_verts")
    if sel["mode"] == "off":
        spatial = np.ones(cat["ra"].size, dtype=bool)
    else:
        has_region = True
        if sel.get("spatial_tool", "ellipse") == "pencil":
            if verts is not None and len(verts) >= 3:
                spatial = polygon_mask(cat["ra"], cat["dec"], verts)
            else:
                spatial = np.ones(cat["ra"].size, dtype=bool)
                has_region = False
        else:
            spatial = ellipse_mask(cat["ra"], cat["dec"], sel["ra_cen"],
                                   sel["dec_cen"], sel["a"], sel["b"],
                                   sel["pa"])
        if sel["mode"] == "outside":
            if has_region:
                spatial = ~spatial
        else:
            if (sel.get("inner_subtract")
                    and sel.get("a_in", 0.0) > 0.0
                    and sel.get("b_in", 0.0) > 0.0):
                spatial = spatial & ~ellipse_mask(
                    cat["ra"], cat["dec"], sel["ra_cen"], sel["dec_cen"],
                    sel["a_in"], sel["b_in"], sel["pa"])
            if (sel.get("pencil_subtract")
                    and sub_verts is not None and len(sub_verts) >= 3):
                spatial = spatial & ~polygon_mask(cat["ra"], cat["dec"],
                                                  sub_verts)
    keep = (spatial
            & (cat["color"] >= sel["color_min"])
            & (cat["color"] <= sel["color_max"])
            & (cat["mag"] >= sel["mag_bright"])
            & (cat["mag"] <= sel["mag_faint"]))
    return spatial, keep


def edge_tip(mag, err, tip_lo=TIP_LO, tip_hi=TIP_HI, n_grid=N_GRID):
    """Continuous edge-detector response and its peak. Returns
    (m_grid, edge, tip); tip is nan when the sample is too thin to calculate (less than 10 samples)."""
    if mag.size < 10:
        return None, None, np.nan
    m_grid = np.linspace(np.floor(mag.min()), np.ceil(mag.max()), n_grid)
    edge = np.array([edge_detect(m, mag, err) for m in m_grid])
    window = (m_grid > tip_lo) & (m_grid < tip_hi)
    if not window.any() or not np.any(edge[window] > 0):
        return m_grid, edge, np.nan
    return m_grid, edge, m_grid[window][np.argmax(edge[window])]


def default_selection(cfg, cat):
    """Starting selection filters from the galaxy's pipeline config: its selection
    ellipse (sky or the sky parameters of a pixel-projected one), its RGB color
    box, and the standard faint cut. Falls back to a circle on the stellar
    density peak (as an initial guess) when no aperture is configured."""
    if "ellipse" in cfg:
        ra_cen, dec_cen, a, b, pa = cfg["ellipse"]
    elif "ellipse_px" in cfg:
        e = cfg["ellipse_px"]
        ra_cen, dec_cen, a, b, pa = (e["ra_cen"], e["dec_cen"],
                                     e["a"], e["b"], e["pa"])
    else:
        density, ra_edges, dec_edges = np.histogram2d(cat["ra"], cat["dec"],
                                                      bins=40)
        i, j = np.unravel_index(np.argmax(density), density.shape)
        ra_cen = 0.5 * (ra_edges[i] + ra_edges[i + 1])
        dec_cen = 0.5 * (dec_edges[j] + dec_edges[j + 1])
        a, b, pa = 30.0, 30.0, 0.0
    color_min, color_max = cfg.get("color", DEFAULT_COLOR)
    return {"ra_cen": ra_cen, "dec_cen": dec_cen, "a": a, "b": b, "pa": pa,
            "mode": "inside", "color_min": color_min, "color_max": color_max,
            "mag_bright": 18.0, "mag_faint": 26.0}