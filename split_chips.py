"""Split a drizzled two-chip FITS mosaic into one file per chip.

The DRC image holds both detector chips in a single array, surrounded by NaNs
and separated by a (diagonal) NaN gap. Connected-component labeling of the
finite pixels finds the two chips; each is cropped to its bounding box, pixels
belonging to the other chip are re-NaNed, and CRPIX1/2 are shifted by the crop
offset so the WCS still maps pixels to the correct sky positions. The WHT and
CTX extensions are cropped identically.

A mosaic drizzled with gap-spanning dithers has NO NaN gap -- the finite
pixels form one connected region and the true chip boundary is gone from the
pixels. Such an image cannot be split (and cannot supply an off-galaxy chip
for background subtraction); split_chips/check_two_chips raise ValueError.

Usage: python split_chips.py <mosaic_drc.fits>
Writes <mosaic_drc>_chip1.fits and <mosaic_drc>_chip2.fits.
"""

import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage


def _find_chips(hdul, path, min_frac=0.25):
    """(labels, [chip1_label, chip2_label]) of the two chips in an open
    mosaic, ordered bottom-to-top on the image. Raises ValueError when the
    image does not hold two cleanly separated comparable chips: a
    single-chip / subarray exposure, or a mosaic whose inter-chip gap was
    filled by drizzling (one connected region). The second-largest
    component must be at least ``min_frac`` of the largest -- anything
    smaller is a stray speck, not a detector chip."""
    name = os.path.basename(path)
    finite = np.isfinite(hdul[1].data)
    labels, n = ndimage.label(finite)
    if n < 2:
        raise ValueError(
            f"{name}: the finite pixels form one connected region -- the "
            "inter-chip gap has been filled by drizzling, so the two "
            "detector chips cannot be identified")
    sizes = ndimage.sum_labels(np.ones_like(labels), labels,
                               index=range(1, n + 1))
    # Two largest components are the chips; anything else is stray specks.
    chip_labels = np.argsort(sizes)[::-1][:2] + 1
    big, small = sizes[chip_labels[0] - 1], sizes[chip_labels[1] - 1]
    if small < min_frac * big:
        raise ValueError(
            f"{name}: second pixel component is only {small / big:.1%} "
            "the size of the largest -- not a two-chip mosaic")
    # Order chip 1/2 bottom-to-top on the image.
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
            for ext in (1, 2, 3):          # SCI, WHT, CTX
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
