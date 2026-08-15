"""Split a drizzled two-chip FITS mosaic into one file per chip. This is used to perform background subtraction by looking at the chip not containing the galaxy.
Usage: python split_chips.py <mosaic_drc.fits>
Writes <mosaic_drc>_chip1.fits and <mosaic_drc>_chip2.fits.
"""

import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage


def _find_chips(hdul, path, min_frac=0.25):
    """Locate the two detector chips inside an open two-chip mosaic.

    A drizzled two-chip image (e.g. ACS/WFC or WFC3/UVIS) covers the sky
    with two rectangular chip footprints separated by a thin gap of blank
    (NaN) pixels where the physical gap between the detectors fell. This
    finds those footprints by labeling the connected regions of finite
    science pixels: each chip shows up as one large connected component,
    and the gap keeps them apart.

    Returns ``(labels, [chip1_label, chip2_label])`` where ``labels`` is the
    full ndimage.label component map of the SCI extension (same shape as the
    image, 0 = blank, k = pixel belongs to component k) and the two label
    values pick out the chips, ordered bottom-to-top on the image so "chip 1"
    and "chip 2" are stable across calls."""

    name = os.path.basename(path)
    # A pixel is part of a chip footprint iff its value is finite;
    # the inter-chip gap and the empty corners around the rotated footprint
    # are NaN. Label every 4-connected island of finite pixels.
    finite = np.isfinite(hdul[1].data)
    labels, n = ndimage.label(finite)
    if n < 2:
        raise ValueError(
            f"{name}: the finite pixels form one connected region -- the "
            "inter-chip gap has been filled by drizzling, so the two "
            "detector chips cannot be identified")
    # Pixel count of every component (labels run 1..n; index k-1 holds k).
    sizes = ndimage.sum_labels(np.ones_like(labels), labels,
                               index=range(1, n + 1))
    # The two largest components are the chip candidates; smaller ones are
    # stray specks (isolated pixels that survived the drizzle edges).
    chip_labels = np.argsort(sizes)[::-1][:2] + 1
    # Real chips are the same physical size, so the runner-up must be at
    # least min_frac of the largest; far smaller means there is only one
    # chip plus debris, not a two-chip mosaic.
    big, small = sizes[chip_labels[0] - 1], sizes[chip_labels[1] - 1]
    if small < min_frac * big:
        raise ValueError(
            f"{name}: second pixel component is only {small / big:.1%} "
            "the size of the largest -- not a two-chip mosaic")
    # Sort the two labels by the top row of each chip's bounding box so
    # chip 1 is always the lower chip on the image, whatever label numbers
    # ndimage happened to assign.
    chip_labels = sorted(
        chip_labels,
        key=lambda lab: ndimage.find_objects(labels == lab)[0][0].start)
    return labels, chip_labels


def check_two_chips(path, min_frac=0.25):
    """Raise ValueError unless ``path`` holds two cleanly separated chips
    (the split_chips criterion), without writing anything."""
    with fits.open(path) as hdul:
        _find_chips(hdul, path, min_frac)


def split_chips(path, min_frac=0.25):
    """Split ``path`` into per-chip files; returns [chip1_path, chip2_path].

    Raises ValueError when the image does not hold two cleanly separated
    chips (see _find_chips).
    """
    with fits.open(path) as hdul:
        labels, chip_labels = _find_chips(hdul, path, min_frac)

        base = os.path.splitext(path)[0]
        out_paths = []
        for i, lab in enumerate(chip_labels, start=1):
            mask = labels == lab
            sl = ndimage.find_objects(mask)[0]

            out = [hdul[0].copy()]
            for ext in (1, 2, 3):
                h = hdul[ext].copy()
                data = h.data[sl].copy()
                # Blank out pixels from the other chip inside the bounding box.
                other = ~mask[sl]
                if np.issubdtype(data.dtype, np.floating):
                    data[other] = np.nan
                else:
                    data[other] = 0
                h.data = data
                # Crop shifts pixel coordinates; move the WCS reference pixel.
                if "CRPIX1" in h.header:
                    h.header["CRPIX1"] -= sl[1].start
                    h.header["CRPIX2"] -= sl[0].start
                out.append(h)

            out_path = f"{base}_chip{i}.fits"
            fits.HDUList(out).writeto(out_path, overwrite=True)
            out_paths.append(out_path)
            print(f"chip {i}: rows {sl[0].start}-{sl[0].stop}, "
                  f"cols {sl[1].start}-{sl[1].stop}, "
                  f"{int(mask.sum())} px -> {out_path}")
    return out_paths


if __name__ == "__main__":
    split_chips(sys.argv[1])
