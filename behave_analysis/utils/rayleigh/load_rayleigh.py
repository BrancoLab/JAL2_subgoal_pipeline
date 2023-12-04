import os

import polars as pl

def load_rayleigh_data(path_to_rayleigh: str) -> pl.DataFrame:
    """Load in Rayleigh data as polars DataFrame"""
    return pl.read_ipc(path_to_rayleigh)

def extract_rayleigh_path(session: object, cluster_type: str, condition: str, file_name: str) -> str:
    """Extract paths to Rayleigh data"""
    path = os.path.join(
        session.base_path,
        session.processed_path,
        "models",
        "Rayleigh",
        cluster_type,
        condition,
        file_name,
    )
    return path