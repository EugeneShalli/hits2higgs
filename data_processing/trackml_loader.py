import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


PAD_TOKEN = 0.0

def load_fatras_data(hits_file, particles_file, truth_file, max_num_hits, normalize=True, chunking=False):
    '''
    Function for reading TrackML .csv files (hits, particles, truth) and creating tensors
    for the hits, track parameters, and particle information.
    
    Parameters:
        hits_file (str): Path to the hits.csv file
        particles_file (str): Path to the particles.csv file
        truth_file (str): Path to the truth.csv file
        max_num_hits (int): Maximum number of hits to pad to
        normalize (bool): Whether to normalize features
        chunking (bool): Whether CSVs should be read in chunks
    '''
    if chunking:
        raise NotImplementedError("Chunking not implemented for this loader.")

    # Load CSVs
    hits = pd.read_csv(hits_file)
    particles = pd.read_csv(particles_file)
    truth = pd.read_csv(truth_file)

    # Merge hits with truth on hit_id
    data = hits.merge(truth, on="hit_id", how="left")
    # Merge with particle properties on particle_id
    data = data.merge(particles[["particle_id", "px", "py", "pz", "q"]], on="particle_id", how="left")

    # Assign synthetic event_id (could be set dynamically in batch loading)
    data["event_id"] = 0

    # Normalize selected columns
    if normalize:
        for col in ["x", "y", "z", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            if std != 0:
                data[col] = (data[col] - mean) / std

    # Shuffle and group by event (here, only one event typically)
    data_grouped_by_event = data.groupby("event_id")

    def extract_hits_data(event_rows):
        coords = event_rows[["x", "y", "z"]].to_numpy(np.float32)
        return np.pad(coords, [(0, max_num_hits - len(coords)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        params = event_rows[["px", "py", "pz", "q"]].to_numpy(np.float32)
        p = np.linalg.norm(params[:, :3], axis=1)
        theta = np.arccos(params[:, 2] / p)
        phi = np.arctan2(params[:, 1], params[:, 0])
        return np.pad(np.column_stack([theta, np.sin(phi), np.cos(phi), params[:, 3]]),
                      [(0, max_num_hits - len(params)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        class_data = event_rows[["particle_id", "weight"]].to_numpy(np.float32)
        return np.pad(class_data, [(0, max_num_hits - len(class_data)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Extract and pad for each event
    hits_data = torch.tensor(np.stack(data_grouped_by_event.apply(extract_hits_data).values))
    track_params_data = torch.tensor(np.stack(data_grouped_by_event.apply(extract_track_params_data).values))
    hit_classes_data = torch.tensor(np.stack(data_grouped_by_event.apply(extract_hit_classes_data).values))

    return hits_data, track_params_data, hit_classes_data

class FatrasTrackDataset(Dataset):
    def __init__(self, base_dir, normalize=True, max_num_hits=None):
        """
        base_dir: path to 'data/' folder
        normalize: whether to normalize features
        max_num_hits: if None, it will be determined automatically
        """
        self.base_dir = Path(base_dir)
        self.normalize = normalize
        self.max_num_hits = max_num_hits
        self.events = []

        for label, class_name in [(1, "signal"), (0, "background")]:
            class_dir = self.base_dir / class_name
            print(f"[INFO] Scanning {class_name} directory...")
            hit_files = list(class_dir.glob("*-hits.csv"))
            event_basenames = sorted(set(
                f.stem.replace("-hits", "")
                for f in tqdm(hit_files, desc=f"Processing {class_name}")
                if (class_dir / f"{f.stem.replace('-hits', '')}-particles.csv").exists() and
                   (class_dir / f"{f.stem.replace('-hits', '')}-truth.csv").exists()
            ))
            self.events.extend([
                {"basename": e, "dir": class_dir, "label": label}
                for e in event_basenames
            ])

        if not self.events:
            raise ValueError("No valid events found in the provided path.")

        np.random.shuffle(self.events)

        if self.max_num_hits is None:
            print("[INFO] Determining max_num_hits automatically...")
            self.max_num_hits = max(
                len(pd.read_csv(event["dir"] / f"{event['basename']}-hits.csv"))
                for event in tqdm(self.events, desc="Counting hits")
            )
            print(f"[INFO] max_num_hits set to {self.max_num_hits}")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        basename = event["basename"]
        event_dir = event["dir"]
        label = event["label"]

        hits = pd.read_csv(event_dir / f"{basename}-hits.csv")
        particles = pd.read_csv(event_dir / f"{basename}-particles.csv")
        truth = pd.read_csv(event_dir / f"{basename}-truth.csv")

        # Merge hits → truth → particles
        data = hits.merge(truth, on="hit_id", how="left")
        data = data.merge(particles[["particle_id", "px", "py", "pz", "q"]], on="particle_id", how="left")

        if self.normalize:
            for col in ["x", "y", "z", "px", "py", "pz", "q"]:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    data[col] = (data[col] - mean) / std

        def extract_hits(event_rows):
            coords = event_rows[["x", "y", "z"]].to_numpy(np.float32)[:self.max_num_hits]
            return np.pad(coords, [(0, self.max_num_hits - len(coords)), (0, 0)], "constant", constant_values=PAD_TOKEN)

        def extract_track_params(event_rows):
            params = event_rows[["px", "py", "pz", "q"]].to_numpy(np.float32)[:self.max_num_hits]
            p = np.linalg.norm(params[:, :3], axis=1)
            theta = np.arccos(np.clip(params[:, 2] / p, -1, 1))
            phi = np.arctan2(params[:, 1], params[:, 0])
            stacked = np.column_stack([theta, np.sin(phi), np.cos(phi), params[:, 3]])
            return np.pad(stacked, [(0, self.max_num_hits - len(stacked)), (0, 0)], "constant", constant_values=PAD_TOKEN)

        def extract_hit_classes(event_rows):
            classes = event_rows[["particle_id", "weight"]].copy()
            classes = classes[:self.max_num_hits]
            return np.pad(classes.to_numpy(np.float32), [(0, self.max_num_hits - len(classes)), (0, 0)], "constant", constant_values=PAD_TOKEN)

        return (
            torch.tensor(extract_hits(data)),
            torch.tensor(extract_track_params(data)),
            torch.tensor(extract_hit_classes(data)),
            torch.tensor(label, dtype=torch.long)
        )