import numpy as np


def from_unit_cube_to_range(z: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return z * (bounds[1] - bounds[0]) + bounds[0]


def from_range_to_unit_cube(z: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return (z - bounds[0]) / (bounds[1] - bounds[0])
