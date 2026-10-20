# HCNS_TRGB

## Install

Requires Python 3.10+.

```sh
git clone https://github.com/HCNS-astro/HCNS_TRGB.git
cd HCNS_TRGB
pip install -r requirements.txt
```

## Run

```sh
python3 trgb_gui.py     
```

## Load Galaxies
To load galaxies, download photometry from the box folder and put them in the galaxies/ folder. 

### macOS app: TODO


## Blind-test mocks (distance accuracy)

`make_mocks.py` builds mock CMDs from a PARSEC population
(`PARSEC_mock_3m_Msol_10Gyr_MH-2.dat`, 3e6 Msol, 10 Gyr, [M/H] = -2):
the population is subsampled to each requested stellar mass, shifted to a
grid of distance moduli, degraded with a real image's completeness and
photometric errors (from its artificial-star tests in
0.5 < F606W-F814W < 1.5), and mixed with the real stars inside a 25"
circle on one half of that image's off-galaxy chip. The other half of the
chip supplies the background sample written into the same file.

```sh
python3 make_mocks.py DW0930P2143 --out mocks/DW0930P2143   # 2 masses x 6 moduli
python3 make_mocks.py NAME --masses 1e6 3e6 --mu 28 30.5 0.5 --out mocks/NAME
```

The image needs its DRC mosaic (`*.fits`, not in git) in `galaxies/<key>/`
so the two chips can be identified. Output: a folder of `mock_NNN.csv`
files (shuffled names, nothing in them but the magnitudes, the image name
and the background scale) that can be handed to a blind tester as is. The
answer sheet (galaxy, mass, m-M, true tip per file) is written outside
that folder, to `mocks/answers/<folder>.csv`; keep it to yourself. The
same thing is available from the GUI toolbar ("Make mocks…").

To fit a mock in the GUI, select its image in the galaxy box, then
"Load CMD…" and pick the file. A CMD-only file has no positions, so the
sky panel and spatial tools are hidden; the fit window defaults to
22-26 mag with the file's background sample subtracted. Any CSV with
`F606W` and `F814W` columns (optionally `e_F814W` and a 0/1 `bg` flag)
loads the same way.

`score_mocks.py` runs that default fit on every mock in a folder and
compares the fitted tip with `tip_true_qt` in the answer sheet (the PARSEC tip
is M = -4.09, not the calibration's -4.015, so compare tips, not moduli):

```sh
python3 score_mocks.py --mocks mocks/DW0930P2143   # -> mocks/answers/DW0930P2143_scores.csv
```
