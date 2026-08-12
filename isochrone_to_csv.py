"""Convert a CMD-web isochrone table (.txt) to CSV.

The CMD web form (http://stev.oapd.inaf.it/cmd) emits a whitespace-separated
table with '#' comment lines and a repeated column-header line before each
isochrone block. This rewrites it as one CSV with a single header row and
every data row from every block, the format of the existing
galaxies/*/isochrone.csv files; blocks stay distinguishable by their MH and
logAge columns. Values are copied verbatim (no float round-trip).

Usage:
    python isochrone_to_csv.py isochrone.txt [output.csv]

The output path defaults to the input with a .csv extension.
"""

import argparse
import os
import sys


def convert(txt_path, csv_path):
    """Write the CSV; returns (n_rows, header columns)."""
    header = None
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                fields = line.lstrip("#").split()
                if "Zini" in fields:
                    if header is not None and fields != header:
                        raise ValueError(
                            f"{txt_path}: blocks have differing columns")
                    header = fields
                continue
            if header is None:
                raise ValueError(f"{txt_path}: data before any header line")
            row = line.split()
            if len(row) != len(header):
                raise ValueError(
                    f"{txt_path}: row with {len(row)} fields, expected "
                    f"{len(header)}: {line[:60]}...")
            rows.append(row)
    if not rows:
        raise ValueError(f"{txt_path}: no data rows found")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    return len(rows), header


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CMD-web isochrone .txt table to CSV.")
    parser.add_argument("txt", help="CMD-web isochrone table")
    parser.add_argument("csv", nargs="?", default=None,
                        help="output CSV (default: input with .csv extension)")
    args = parser.parse_args()

    csv_path = args.csv or os.path.splitext(args.txt)[0] + ".csv"
    try:
        n, header = convert(args.txt, csv_path)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    print(f"{csv_path}: {n} rows, {len(header)} columns")


if __name__ == "__main__":
    main()
