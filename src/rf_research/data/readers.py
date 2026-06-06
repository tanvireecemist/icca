from __future__ import annotations

from pathlib import Path

import numpy as np


def _as_complex(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array.reshape(-1).astype(np.complex64, copy=False)
    array = np.squeeze(array)
    if array.ndim == 2 and array.shape[0] == 2:
        return (array[0] + 1j * array[1]).astype(np.complex64)
    if array.ndim == 2 and array.shape[-1] == 2:
        return (array[:, 0] + 1j * array[:, 1]).astype(np.complex64)
    flat = array.reshape(-1)
    if flat.size % 2:
        flat = flat[:-1]
    return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        keys = list(bundle.files)
        for in_phase, quadrature in (
            ("beaconPreambleI", "beaconPreambleQ"),
            ("beaconDataI", "beaconDataQ"),
            ("I", "Q"),
            ("i", "q"),
        ):
            if in_phase in bundle and quadrature in bundle:
                i_value = np.asarray(bundle[in_phase]).reshape(-1)
                q_value = np.asarray(bundle[quadrature]).reshape(-1)
                count = min(i_value.size, q_value.size)
                return (i_value[:count] + 1j * q_value[:count]).astype(np.complex64)
        preferred = ["iq", "samples", "signal", "data", "arr_0"]
        for key in preferred + keys:
            if key not in bundle:
                continue
            array = np.asarray(bundle[key])
            if np.issubdtype(array.dtype, np.number) and array.size >= 2:
                return _as_complex(array)
    raise ValueError(f"No numeric signal array found in {path}")


def load_iq(
    path: str | Path,
    dtype: str,
    offset: int,
    n_samples: int,
    scale: float,
) -> np.ndarray:
    signal_path = Path(path)
    if dtype == "sc16":
        available = max(0, signal_path.stat().st_size // 4 - int(offset))
        read_count = min(int(n_samples), available)
        raw = np.memmap(
            signal_path,
            dtype="<i2",
            mode="r",
            offset=int(offset) * 4,
            shape=(read_count * 2,),
        )
        iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    elif dtype == "complex64":
        available = max(0, signal_path.stat().st_size // 8 - int(offset))
        read_count = min(int(n_samples), available)
        iq = np.memmap(
            signal_path,
            dtype="<c8",
            mode="r",
            offset=int(offset) * 8,
            shape=(read_count,),
        ).astype(np.complex64)
    elif dtype == "npz":
        iq = _load_npz(signal_path)
        start = int(offset)
        iq = iq[start : start + int(n_samples)]
    else:
        raise ValueError(f"Unsupported signal dtype: {dtype}")

    iq = np.asarray(iq, dtype=np.complex64) / float(scale)
    if iq.size < n_samples:
        iq = np.pad(iq, (0, int(n_samples) - iq.size))
    elif iq.size > n_samples:
        start = (iq.size - int(n_samples)) // 2
        iq = iq[start : start + int(n_samples)]

    iq = iq - np.mean(iq)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2) + 1e-8))
    iq = iq / max(rms, 1e-4)
    return np.stack((iq.real, iq.imag), axis=0).astype(np.float32, copy=False)
