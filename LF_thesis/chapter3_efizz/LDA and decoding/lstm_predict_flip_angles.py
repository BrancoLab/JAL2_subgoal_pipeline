"""Run LSTM decoders to predict pre/post flip A tuning angles per condition."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_11thSept, JAL4_19thSept, JAL4_28aug, JAL4_3rdSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_21stSept, JAL005_8thSept
from behave_analysis.database.Experiments.JAL006_ex import (
    JAL6_28mar,
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_flip7_1apr,
)
from behave_analysis.database.Experiments.JAL007_ex import (
    JAL7_23apr,
    JAL7_30apr,
    JAL7_flip2_12mar,
    JAL7_flip5_22mar,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
)
from behave_analysis.database.Experiments.JAL008_ex import (
    JAL8_14may,
    JAL8_21may,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_tiny_3may,
)
from behave_analysis.process.session import get_experiment
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        if self.device == "cuda":
            logger.info("Using GPU for LSTM.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return torch.tanh(out) * np.pi


def reshape_sequences_1d(
    X: np.ndarray,
    Y: np.ndarray,
    seq_length: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_sequences = len(X) - seq_length + 1
    if num_sequences <= 1:
        raise ValueError("Not enough samples for the desired sequence length.")

    X_seq = np.zeros((num_sequences, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(num_sequences, dtype=np.float32)
    for i in range(num_sequences):
        X_seq[i] = X[i : i + seq_length]
        y_seq[i] = Y[i + seq_length - 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=test_size,
        random_state=random_state,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.float32).to(device),
    )


def run_lstm(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 500,
    seq_len: int = 5,
    hidden_dim: int = 128,
    learning_rate: float = 0.005,
    verbose: bool = True,
) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, X_test, y_train, y_test = reshape_sequences_1d(X, y, seq_len)

    model = LSTMModel(X.shape[1], hidden_dim, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        with autocast():
            preds = model(X_train).squeeze()
            loss = loss_fn(preds, y_train)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if verbose and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test).squeeze()
                test_loss = loss_fn(test_preds, y_test)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            logger.info(f"Epoch {epoch}: train loss {loss.item():.4f}, test loss {test_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu().numpy().squeeze()
        test_preds = model(X_test).cpu().numpy().squeeze()

    train_r2 = r2_score(y_train.cpu().numpy(), train_preds)
    test_r2 = r2_score(y_test.cpu().numpy(), test_preds)

    if 0:
        epochs_axis = np.arange(0, num_epochs, 50)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(epochs_axis[: len(train_losses)], train_losses, label="Train loss")
        plt.plot(epochs_axis[: len(test_losses)], test_losses, label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Learning curve")

        plt.subplot(2, 2, 3)
        plt.plot(train_preds[:500], label="Predicted")
        plt.plot(y_train.cpu().numpy()[:500], label="Actual")
        plt.title(f"Train predictions (R²={train_r2:.3f})")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(test_preds[:500], label="Predicted")
        plt.plot(y_test.cpu().numpy()[:500], label="Actual")
        plt.title(f"Test predictions (R²={test_r2:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"train_r2": train_r2, "test_r2": test_r2}


def load_session_data(session) -> Tuple[np.ndarray, pl.DataFrame]:
    video_df = pl.read_csv(Path(session.base_path) / session.processed_path / "full_video_dataframe.csv")
    cluster_type = "good"
    matrix_path = Path(session.base_path) / session.processed_path / f"frame_by_{cluster_type}_cluster_matrix.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(f"{matrix_path} not found.")
    matrix = np.load(matrix_path)
    return matrix, video_df


def extract_angle_targets(video_df: pl.DataFrame, condition: str, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    df_cond = video_df.filter(pl.col("condition") == condition)
    if df_cond.is_empty():
        raise ValueError(f"No frames for condition {condition}")
    Y = df_cond[target_cols].to_numpy()
    return df_cond, Y


def run_session_models(session, session_name: str):
    # try to load data if data load fails skip this function
    try:
        matrix, video_df = load_session_data(session)
        results = []

        for condition, target_col in [
            ("barrier_pre_flip", "h_preflipbar_a"),
            ("barrier_pre_flip", "h_postflipbar_a"),
            ("barrier_post_flip", "h_preflipbar_a"),
            ("barrier_post_flip", "h_postflipbar_a"),
        ]:
            if target_col not in video_df.columns:
                logger.warning(f"{target_col} not found in video dataframe.")
                continue

            df_cond = filter_video_dataframe(video_df, condition)

            if df_cond.is_empty():
                continue

            idx = df_cond["frames"].to_numpy().astype(int)  # get the frame indices for this condition
            print(idx)
            print(idx.max(), matrix.shape[0])

            if idx.max() >= matrix.shape[0]:
                # remove the last index if it exceeds matrix size
                idx = idx[idx < matrix.shape[0]]

            X_cond = matrix[idx]  # select the rows of the matrix corresponding to these frames

            print(X_cond)
            Y_cond = df_cond[target_col].to_numpy().astype(np.float32)  # get the target angles in the condition

            logger.info(f"{session_name}: LSTM predicting {target_col} for {condition} with {len(Y_cond)} samples.")
            try:
                metrics = run_lstm(X_cond, Y_cond, verbose=True)
                logger.success(f"{session_name} condition {condition} target {target_col}: " f"train R²={metrics['train_r2']:.3f}, test R²={metrics['test_r2']:.3f}")
            except Exception as exc:
                logger.error(f"Failed {session_name} {condition} {target_col}: {exc}")
                metrics = {"train_r2": np.nan, "test_r2": np.nan}
            results.append((session_name, condition, target_col, metrics["test_r2"]))

    except Exception as e:
        logger.error(f"Skipping {session_name} due to data loading error: {e}")
        results = []
    return results


def main():
    sessions = [
        JAL6_flip7_1apr,
        JAL6_flip3_18mar,
        JAL6_flip4_21mar,
        JAL6_flip5_25mar,
        JAL6_28mar,
        JAL3_25aug,
        JAL3_1sept,
        JAL3_4sept,
        JAL3_7sept,
        JAL005_8thSept,
        JAL005_21stSept,
        JAL7_sesh8_9apr,
        JAL7_sesh9_16apr,
        JAL7_flip5_22mar,
        JAL7_flip2_12mar,
        JAL7_23apr,
        JAL8_flip1_25apr,
        JAL8_flip2_29apr,
        JAL8_flip4_10may,
        JAL8_14may,
        JAL8_21may,
        JAL4_3rdSept,
        JAL4_19thSept,
        JAL4_28aug,
        JAL4_11thSept,
    ]

    session_NAMES = [
        "JAL6_flip7_1apr",
        "JAL6_flip3_18mar",
        "JAL6_flip4_21mar",
        "JAL6_flip5_25mar",
        "JAL6_28mar",
        "JAL3_25aug",
        "JAL3_1sept",
        "JAL3_4sept",
        "JAL3_7sept",
        "JAL005_8thSept",
        "JAL005_21stSept",
        "JAL7_sesh8_9apr",
        "JAL7_sesh9_16apr",
        "JAL7_flip5_22mar",
        "JAL7_flip2_12mar",
        "JAL7_23apr",
        "JAL8_flip1_25apr",
        "JAL8_flip2_29apr",
        "JAL8_flip4_10may",
        "JAL8_14may",
        "JAL4_3rdSept",
        "JAL4_19thSept",
        "JAL4_28aug",
        "JAL4_11thSept",
    ]
    all_results = []
    for session_obj, session_name in zip(sessions, session_NAMES):
        session = get_experiment(session_obj)
        logger.info(f"Processing session: {session_name}")
        try:
            session_results = run_session_models(session, session_name)
            all_results.extend(session_results)
        except Exception as e:
            logger.error(f"Skipping {session_name} due to {e}")

    print("\nSummary:")
    for sess, cond, target, r2 in all_results:
        print(f"{sess} | {cond} | {target}: test R²={r2:.3f}")

    # save the results as a pickle file
    out_path = r"Z:\Laurence\thesis\efizz_chapter\LSTM_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    main()
