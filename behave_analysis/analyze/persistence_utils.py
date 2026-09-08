"""Shared persistence helpers for analysis result artifacts.

Supports:
- HDF5 save/load for nested dictionaries
- NPZ save/load with legacy normalization
- JSON save for settings/metadata
"""

import json
import os
from typing import Any, Dict

import h5py
import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert numpy types into JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save settings/metadata as readable JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_jsonable)


def _write_hdf5_group(h5_group: h5py.Group, data_dict: Dict[str, Any]) -> None:
    for key, val in data_dict.items():
        key = str(key)

        if isinstance(val, dict):
            sub = h5_group.create_group(key)
            _write_hdf5_group(sub, val)
            continue

        if val is None:
            ds = h5_group.create_dataset(key, data=np.array([], dtype=np.float32))
            ds.attrs["is_none"] = True
            continue

        if isinstance(val, str):
            ds = h5_group.create_dataset(key, data=np.array(val, dtype=h5py.string_dtype("utf-8")))
            ds.attrs["is_str"] = True
            continue

        arr = np.asarray(val)

        if arr.dtype == object:
            # Object arrays cannot be stored natively in HDF5 in a portable way.
            # Store a JSON-serialized representation.
            ds = h5_group.create_dataset(key, data=np.array(json.dumps(_to_jsonable(val)), dtype=h5py.string_dtype("utf-8")))
            ds.attrs["is_json"] = True
            continue

        if arr.ndim > 0 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            h5_group.create_dataset(key, data=arr, compression="gzip")
        else:
            h5_group.create_dataset(key, data=arr)


def save_hdf5(path: str, data_dict: Dict[str, Any], overwrite: bool = True) -> str:
    """Save nested dict to HDF5 hierarchy."""
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"HDF5 already exists: {path}")
    with h5py.File(path, "w") as h5f:
        _write_hdf5_group(h5f, data_dict)
    return path


def _read_hdf5_group(h5_group: h5py.Group) -> Dict[str, Any]:
    out = {}
    for key, obj in h5_group.items():
        if isinstance(obj, h5py.Group):
            out[key] = _read_hdf5_group(obj)
            continue

        if obj.attrs.get("is_none", False):
            out[key] = None
            continue

        val = obj[()]

        if obj.attrs.get("is_json", False):
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            out[key] = json.loads(val)
            continue

        if isinstance(val, bytes):
            out[key] = val.decode("utf-8")
        elif isinstance(val, np.ndarray) and val.shape == ():
            out[key] = val.item()
        else:
            out[key] = val

    return out


def load_hdf5(path: str) -> Dict[str, Any]:
    """Load nested dict from HDF5."""
    with h5py.File(path, "r") as h5f:
        return _read_hdf5_group(h5f)


def save_npz(path: str, data_dict: Dict[str, Any]) -> str:
    """Save dict to NPZ (legacy-compatible)."""
    np.savez(path, **data_dict)
    return path


def normalize_legacy_npz(npz_obj, drop_keys=("allow_pickle",)) -> Dict[str, Any]:
    """Convert np.load(npz) output into plain dict and unwrap 0-d object arrays."""
    out = {}
    for key in npz_obj.files:
        if key in drop_keys:
            continue
        val = npz_obj[key]
        if isinstance(val, np.ndarray) and val.dtype == object and val.shape == ():
            out[key] = val.item()
        else:
            out[key] = val
    return out


def load_npz(path: str, allow_pickle: bool = True, drop_keys=("allow_pickle",)) -> Dict[str, Any]:
    """Load NPZ into dict with legacy normalization."""
    with np.load(path, allow_pickle=allow_pickle) as npz_obj:
        return normalize_legacy_npz(npz_obj, drop_keys=drop_keys)


def load_results_with_fallback(h5_path: str, npz_path: str, prefer_hdf5: bool = True, drop_keys=("allow_pickle",)) -> Dict[str, Any]:
    """Load results preferring HDF5, then NPZ, and return a plain dict."""
    if prefer_hdf5 and os.path.exists(h5_path):
        return load_hdf5(h5_path)

    if os.path.exists(npz_path):
        return load_npz(npz_path, allow_pickle=True, drop_keys=drop_keys)

    if os.path.exists(h5_path):
        return load_hdf5(h5_path)

    raise FileNotFoundError(f"No results found. Tried: {h5_path} and {npz_path}")


def convert_npz_to_hdf5(npz_path: str, h5_path: str, overwrite: bool = False, drop_keys=("allow_pickle",)) -> Dict[str, Any]:
    """Convert legacy NPZ to HDF5 and return normalized dict."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Legacy NPZ not found: {npz_path}")
    data = load_npz(npz_path, allow_pickle=True, drop_keys=drop_keys)
    save_hdf5(h5_path, data_dict=data, overwrite=overwrite)
    return data
