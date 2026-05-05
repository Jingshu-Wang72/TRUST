import numpy as np


DEFAULT_VIEWS = (
    "raw,slope,trend,residual,delta,curvature,wavelet,"
    "frequency_energy,curvature_event,fft,acf,paa,minirocket,multirocket,hydra"
)

VIEW_ALIASES = {
    "frequency-energy": "frequency_energy",
    "freq_energy": "frequency_energy",
    "frequencyenergy": "frequency_energy",
    "curvature-event": "curvature_event",
    "curv_event": "curvature_event",
    "fft": "fft",
    "acf": "acf",
    "paa": "paa",
    "minirocket": "minirocket",
    "mini_rocket": "minirocket",
    "MiniROCKET".lower(): "minirocket",
    "multirocket": "multirocket",
    "multi_rocket": "multirocket",
    "MultiROCKET".lower(): "multirocket",
    "hydra": "hydra",
    "HYDRA".lower(): "hydra",
}


def parse_views(view_spec: str) -> list[str]:
    views = []
    for name in view_spec.split(","):
        key = name.strip()
        if not key:
            continue
        views.append(VIEW_ALIASES.get(key.lower(), key))
    return views


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32, copy=True)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded).astype(np.float32)


def acf_features(x: np.ndarray, max_lag: int = 32) -> np.ndarray:
    centered = x - np.mean(x, axis=1, keepdims=True)
    denom = np.sum(centered * centered, axis=1, keepdims=True) + 1e-10
    feats = []
    max_lag = min(max_lag, x.shape[1] - 1)
    for lag in range(1, max_lag + 1):
        num = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1, keepdims=True)
        feats.append(num / denom)
    if not feats:
        return np.zeros((x.shape[0], 1), dtype=np.float32)
    return np.concatenate(feats, axis=1).astype(np.float32)


def wavelet_features(x: np.ndarray, widths: tuple[int, ...] = (2, 4, 8, 16)) -> np.ndarray:
    feats = []
    for width in widths:
        kernel = ricker(points=max(3, width * 4), width=width)
        conv = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, x)
        feats.extend([
            np.mean(conv, axis=1, keepdims=True),
            np.std(conv, axis=1, keepdims=True),
            np.max(conv, axis=1, keepdims=True),
            np.min(conv, axis=1, keepdims=True),
        ])
    return np.concatenate(feats, axis=1).astype(np.float32)


def ricker(points: int, width: float) -> np.ndarray:
    radius = (points - 1) / 2.0
    xs = np.arange(points, dtype=np.float32) - radius
    wsq = np.float32(width * width)
    factor = np.float32(2.0 / (np.sqrt(3.0 * width) * np.pi ** 0.25))
    kernel = factor * (1.0 - (xs * xs) / wsq) * np.exp(-(xs * xs) / (2.0 * wsq))
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel = kernel / norm
    return kernel.astype(np.float32)


def freq_band_energy(x: np.ndarray, bands: int = 8) -> np.ndarray:
    fft_mag = np.abs(np.fft.rfft(x, axis=1)).astype(np.float32)
    power = fft_mag ** 2
    chunks = np.array_split(power, bands, axis=1)
    band_energy = [np.sum(chunk, axis=1, keepdims=True) for chunk in chunks]
    total = np.sum(power, axis=1, keepdims=True) + 1e-10
    return (np.concatenate(band_energy, axis=1) / total).astype(np.float32)


def fft_features(x: np.ndarray, n_bins: int = 128) -> np.ndarray:
    fft_mag = np.abs(np.fft.rfft(x, axis=1)).astype(np.float32)
    if fft_mag.shape[1] == n_bins:
        return fft_mag
    if fft_mag.shape[1] > n_bins:
        return fft_mag[:, :n_bins].astype(np.float32)
    pad_width = n_bins - fft_mag.shape[1]
    return np.pad(fft_mag, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)


def paa_stats(x: np.ndarray, segments: int = 16) -> np.ndarray:
    chunks = np.array_split(x, segments, axis=1)
    means, stds = [], []
    for chunk in chunks:
        means.append(np.mean(chunk, axis=1, keepdims=True))
        stds.append(np.std(chunk, axis=1, keepdims=True))
    return np.concatenate(means + stds, axis=1).astype(np.float32)


def curvature_event_hist(curvature: np.ndarray, bins: int = 5) -> np.ndarray:
    feats = []
    for row in curvature:
        if np.allclose(row, row[0]):
            hist = np.zeros(bins, dtype=np.float32)
            hist[0] = 1.0
        else:
            hist, _ = np.histogram(row, bins=bins, density=False)
            hist = hist.astype(np.float32) / max(1, row.shape[0])
        feats.append(hist)
    return np.vstack(feats).astype(np.float32)


def minirocket_features(
    x_fit: np.ndarray,
    x_transform: np.ndarray,
    num_kernels: int = 9996,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from sktime.transformations.panel.rocket import MiniRocket
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MiniROCKET view requires sktime. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    transformer = MiniRocket(num_kernels=int(num_kernels), random_state=int(random_state))
    transformer.fit(x_fit[:, np.newaxis, :].astype(np.float32))
    features = transformer.transform(x_transform[:, np.newaxis, :].astype(np.float32))
    if hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float32)


def multirocket_features(
    x_fit: np.ndarray,
    x_transform: np.ndarray,
    num_kernels: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from sktime.transformations.panel.rocket import MultiRocket
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MultiROCKET view requires sktime. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    transformer = MultiRocket(num_kernels=int(num_kernels), random_state=int(random_state))
    transformer.fit(x_fit[:, np.newaxis, :].astype(np.float32))
    features = transformer.transform(x_transform[:, np.newaxis, :].astype(np.float32))
    if hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float32)


def hydra_features(
    x_fit: np.ndarray,
    x_transform: np.ndarray,
    num_kernels: int = 8,
    n_groups: int = 64,
    max_num_channels: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from aeon.transformations.collection.convolution_based import HydraTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HYDRA view requires aeon. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    kwargs = {
        "n_kernels": int(num_kernels),
        "n_groups": int(n_groups),
        "max_num_channels": int(max_num_channels),
        "random_state": int(random_state),
    }
    try:
        transformer = HydraTransformer(output_type="numpy", **kwargs)
    except TypeError:
        transformer = HydraTransformer(**kwargs)
    transformer.fit(x_fit[:, np.newaxis, :].astype(np.float32))
    features = transformer.transform(x_transform[:, np.newaxis, :].astype(np.float32))
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    elif hasattr(features, "to_numpy"):
        features = features.to_numpy()
    return np.asarray(features, dtype=np.float32)


def fit_view_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_view(x: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    if stats is None:
        stats = fit_view_scaler(x)
    mean, std = stats
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def build_views(
    x: np.ndarray,
    view_names: list[str],
    trend_window: int = 5,
    x_fit: np.ndarray | None = None,
    minirocket_num_kernels: int = 9996,
    minirocket_random_state: int = 42,
    multirocket_num_kernels: int = 1000,
    multirocket_random_state: int = 42,
    hydra_num_kernels: int = 8,
    hydra_n_groups: int = 64,
    hydra_max_num_channels: int = 8,
    hydra_random_state: int = 42,
    view_stats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    view_names = [VIEW_ALIASES.get(name.lower(), name) for name in view_names]
    raw = x.astype(np.float32)
    slope = np.gradient(raw, axis=1).astype(np.float32)
    trend = moving_average(raw, trend_window)
    residual = (raw - trend).astype(np.float32)
    delta = np.diff(raw, axis=1, prepend=raw[:, :1]).astype(np.float32)
    curvature = np.gradient(slope, axis=1).astype(np.float32)

    available = {
        "raw": raw,
        "slope": slope,
        "trend": trend,
        "residual": residual,
        "delta": delta,
        "curvature": curvature,
        "acf": acf_features(raw),
        "wavelet": wavelet_features(raw),
        "frequency_energy": freq_band_energy(raw),
        "fft": fft_features(raw),
        "paa": paa_stats(raw),
        "curvature_event": curvature_event_hist(curvature),
    }
    if "minirocket" in view_names:
        if x_fit is None:
            x_fit = raw
        available["minirocket"] = minirocket_features(
            x_fit=x_fit,
            x_transform=raw,
            num_kernels=minirocket_num_kernels,
            random_state=minirocket_random_state,
        )
    if "multirocket" in view_names:
        if x_fit is None:
            x_fit = raw
        available["multirocket"] = multirocket_features(
            x_fit=x_fit,
            x_transform=raw,
            num_kernels=multirocket_num_kernels,
            random_state=multirocket_random_state,
        )
    if "hydra" in view_names:
        if x_fit is None:
            x_fit = raw
        available["hydra"] = hydra_features(
            x_fit=x_fit,
            x_transform=raw,
            num_kernels=hydra_num_kernels,
            n_groups=hydra_n_groups,
            max_num_channels=hydra_max_num_channels,
            random_state=hydra_random_state,
        )
    missing = [name for name in view_names if name not in available]
    if missing:
        raise ValueError(f"Unknown view(s): {missing}. Available views: {sorted(available)}")
    fitted_stats = {} if view_stats is None else dict(view_stats)
    standardized = {}
    for name in view_names:
        if name not in fitted_stats:
            fitted_stats[name] = fit_view_scaler(available[name])
        standardized[name] = standardize_view(available[name], fitted_stats[name])
    return standardized, fitted_stats
