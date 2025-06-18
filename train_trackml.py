import torch
import torch.nn as nn
import numpy as np
import argparse
import wandb

from torch.utils.data import DataLoader, random_split

from model import TransformerRegressor, save_model
from data_processing.trackml_loader import FatrasTrackDataset, PAD_TOKEN
from evaluation.scoring import calc_score_trackml
from hdbscan import HDBSCAN


import os
os.environ["WANDB_API_KEY"] = "226997cfa718d7920cc402f08b0a8035f0c25fe1"  # 👈 Replace with your actual API key
os.environ["WANDB_MODE"] = "online"  # or "offline" if no internet access



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clustering(pred_params, min_cl_size, min_samples):
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []
    for event_prediction in pred_params:
        regressed_params = np.array(event_prediction.tolist())
        event_cluster_labels = clustering_algorithm.fit_predict(regressed_params)
        cluster_labels.append(torch.from_numpy(event_cluster_labels).int())
    return cluster_labels


def train_epoch(model, optim, train_loader, loss_fn):
    model.train()
    total_loss = 0.

    for hits, track_params, _, _ in train_loader:
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        optim.zero_grad()

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)

        loss = loss_fn(pred, track_params)
        loss.backward()
        optim.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for hits, track_params, _, _ in val_loader:
            hits = hits.to(DEVICE)
            track_params = track_params.to(DEVICE)
            padding_mask = (hits == PAD_TOKEN).all(dim=2)

            pred = model(hits, padding_mask)
            pred = torch.unsqueeze(pred[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)

            loss = loss_fn(pred, track_params)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main(args):
    # wandb.init(project="trackml-fatras", config=vars(args))
    wandb.init(
    project="trackml-fatras",
    name=f"run_{args.model_name}",
    config=vars(args),
    # entity="your_wandb_username_or_team"
    )


    dataset = FatrasTrackDataset(args.data_path, normalize=True, max_num_hits=args.max_nr_hits)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    model = TransformerRegressor(
        num_encoder_layers=args.nr_enc_layers,
        d_model=args.embedding_size,
        n_head=args.nr_heads,
        input_size=3,
        output_size=4,
        dim_feedforward=args.hidden_dim,
        dropout=args.dropout
    ).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = np.inf
    train_losses, val_losses = [], []
    patience = 0

    for epoch in range(args.nr_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, "best", val_losses, train_losses, epoch, patience, args.model_name)
            patience = 0
        else:
            save_model(model, optimizer, "last", val_losses, train_losses, epoch, patience, args.model_name)
            patience += 1

        if patience >= args.early_stop:
            print("Early stopping...")
            break

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_epochs', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--max_nr_hits', type=int, default=30000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)

    parser.add_argument('--nr_enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--nr_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
