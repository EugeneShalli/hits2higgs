#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from tqdm import tqdm
# from tqdm.notebook import tqdm
import argparse
import acts
import acts.examples
from acts.examples import DigitizationCoordinatesConverter, readDigiConfigFromJson

def convert_digitized_hits_csv(csv_file: Path, converter: DigitizationCoordinatesConverter):
    print(f"[INFO] Converting: {csv_file}")

    # Load measurement data
    df = pd.read_csv(csv_file)

    # Convert all local coordinates to global
    global_coords = []
    volumes = []
    layers = []
    modules = []

    for row in df.itertuples(index=False):
        try:
            geo_id = acts.GeometryIdentifier(int(row.geometry_id))
            volumes.append(geo_id.volume)
            layers.append(geo_id.layer)
            modules.append(geo_id.sensitive)

            g = converter.localToGlobal(int(row.geometry_id), float(row.local0), float(row.local1))
        except RuntimeError:
            g = (float("nan"), float("nan"), float("nan"))
            volumes.append(-1)
            layers.append(-1)
            modules.append(-1)

        global_coords.append(g)

    x, y, z = zip(*global_coords)
    df["global_x"] = x
    df["global_y"] = y
    df["global_z"] = z
    df["volume"] = volumes
    df["layer"] = layers
    df["module"] = modules

    # Write to CSV
    out_csv = csv_file.with_name(csv_file.stem + "_global_xyz.csv")
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved: {out_csv}")

def find_measurement_files(root_dir: Path):
    """
    Recursively find all 'measurements.csv' files in the directory structure.
    """
    files = list(root_dir.rglob("*-measurements.csv"))
    return [f for f in files if f.is_file()]

def main(dataset_dir: Path, digi_config_file: Path):
    # Build ACTS geometry once
    print(f"[INFO] Building tracking geometry...")
    detector = acts.examples.GenericDetector()
    tracking_geometry = detector.trackingGeometry()
    print(f"[✓] Tracking geometry built.")

    # Build Digitization config for the coordinate converter once
    digi_config = acts.examples.DigitizationAlgorithm.Config(
        digitizationConfigs=readDigiConfigFromJson(str(digi_config_file.resolve())),
        surfaceByIdentifier=tracking_geometry.geoIdSurfaceMap()
    )
    converter = DigitizationCoordinatesConverter(digi_config)

    # Find measurement files
    measurement_files = []
    for subfolder in ['background', 'signal']:
        digitization_dir = dataset_dir / subfolder / 'digitization'
        if digitization_dir.exists():
            measurement_files.extend(find_measurement_files(digitization_dir))

    print(f"[INFO] Found {len(measurement_files)} measurement files.")

    for csv_file in tqdm(measurement_files, desc="Digitizing Files"):
        convert_digitized_hits_csv(csv_file, converter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digitize all measurement CSV files in the dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the root dataset directory.")
    parser.add_argument("digi_config", type=Path, help="Path to the digitization config JSON file.")
    args = parser.parse_args()

    main(args.dataset_dir, args.digi_config)
