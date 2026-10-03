import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    singular_values = np.linalg.svd(delta_W, compute_uv=False)

    energy = singular_values ** 2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return 0

    cumulative_energy = np.cumsum(energy)
    energy_ratio = cumulative_energy / total_energy

    return np.searchsorted(energy_ratio, energy_threshold) + 1