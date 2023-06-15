# do not evaluate annotations at runtime
from __future__ import annotations

import numpy as np
import math
from functools import cached_property
from scipy.spatial import ConvexHull, Delaunay


def hexagonal_grid_in_circle(spacing: float, radius: float) -> np.ndarray:
    points = set()

    # Define a helper function to check if a point is within the circle
    def is_point_in_circle(x, y):
        return math.sqrt(x**2 + y**2) <= radius

    # Generate points in a hexagonal grid centered at (0,0)
    for i in range(-int(radius / spacing) - 2, int(radius / spacing) + 2):
        for j in range(-int(radius / spacing) - 2, int(radius / spacing) + 2):
            # offset every other row to make a hexagonal grid
            x = i * spacing + (j % 2) * math.sqrt(3) / 4
            y = j * spacing
            # If the point is within the circle, add it to the set
            if is_point_in_circle(x, y):
                points.add((x, y))

    return np.asarray(list(points))


class Geometry:
    def __init__(self, spacing: float, radius: float, jitter=None):
        self._spacing = spacing
        self.xy = hexagonal_grid_in_circle(spacing, radius)
        if jitter:
            self.xy += np.random.multivariate_normal(
                np.zeros(2), jitter**2 * np.eye(2), size=self.xy.shape[0]
            )

        x = np.max(self.xy[:, 0])
        scale = (x - self._spacing / 2) / x
        self.fiducial_hull = ConvexHull(self.xy * scale)

        self.fiducial_triangulation = Delaunay(
            self.fiducial_hull.points[self.fiducial_hull.vertices]
        )

        mask = self.in_fiducial_area(self.xy)
        # sensors in the fiducial region
        self.fiducial_xy = self.xy[mask]
        # sensors in the veto region
        self.veto_xy = self.xy[~mask]

    @cached_property
    def fiducial_boundary(self) -> np.ndarray:
        return np.concatenate(
            [
                self.fiducial_hull.points[self.fiducial_hull.vertices],
                [self.fiducial_hull.points[self.fiducial_hull.vertices][0]],
            ]
        )

    def in_fiducial_area(self, xy):
        return self.fiducial_triangulation.find_simplex(xy) >= 0


class LightYield:
    """
    Light yields from https://arxiv.org/pdf/1311.4767.pdf eq. 5
    """

    l_a = 24 / 125
    l_e = 98 / 125
    l_p = np.sqrt(l_a * l_e / 3)
    l_c = l_e / (3 * np.exp(-l_e / l_a))
    sin_tc = np.sin(np.radians(41))
    sqrt_lmu = l_c / sin_tc * np.sqrt(2 / (np.pi * l_p))

    @classmethod
    def cascade(cls, rx):
        r = np.maximum(1e-2, rx)
        return np.exp(-r / cls.l_p) / (4 * np.pi * cls.l_c * r * np.tanh(r / cls.l_c))

    @classmethod
    def track(cls, rx):
        r = np.maximum(1e-2, rx)
        return np.exp(-r / cls.l_p) / (
            2
            * np.pi
            * cls.sin_tc
            * np.sqrt(r)
            * cls.sqrt_lmu
            * np.tanh(np.sqrt(r) / cls.sqrt_lmu)
        )


def sample_trajectories(radius, size=1000):
    """
    Return points where isotropic rays enter and exit a circle
    """
    # sample uniformly in direction
    angle = np.random.uniform(0, 2 * np.pi, size=size)
    # sample uniformly in impact parameter
    impact = np.random.uniform(-radius, radius, size=size)[..., None]
    # direction of ray
    d = np.stack((-np.cos(angle), -np.sin(angle)), axis=-1)
    # point of closest aproach
    closest_approach = impact * np.stack((-d[..., 1], d[..., 0]), axis=-1)
    # distance to entry and exit points
    half_chord = np.sqrt(radius**2 - impact**2)
    return closest_approach - d * half_chord, closest_approach + d * half_chord


def simulate_events(
    geometry: Geometry,
    injection_radius: float,
    angle: np.ndarray,
    impact: np.ndarray,
    log_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # direction of ray
    d = np.stack((-np.cos(angle), -np.sin(angle)), axis=-1)
    # point of closest aproach
    closest_approach = impact[..., None] * np.stack((-d[..., 1], d[..., 0]), axis=-1)
    # distance to entry and exit points
    half_chord = np.sqrt(injection_radius**2 - impact[..., None] ** 2)

    entry = closest_approach - d * half_chord
    exit = closest_approach + d * half_chord
    loss_point = (
        closest_approach
        + d
        * half_chord
        * np.random.uniform(-1, 1, size=angle.size).reshape(angle.shape)[..., None]
    )
    d_loss = point_point_distances(geometry.fiducial_xy, loss_point)
    d_track = point_line_distances(
        geometry.veto_xy,
        entry,
        exit,
    )

    energy = 10**log_energy
    # amplitude of largest energy loss
    loss = energy * (1 - np.random.uniform(0, 1, size=energy.shape) ** (1.0 / 5))

    # signals in fiducial and veto region
    fiducial_signal = np.random.poisson(loss[..., None] * LightYield.cascade(d_loss))
    veto_signal = np.random.poisson(
        1e-3 * energy[..., None] * LightYield.track(d_track)
    )

    return entry, exit, loss_point, fiducial_signal, veto_signal


def classify_events(
    geometry: Geometry,
    loss_point: np.ndarray,
    fiducial_signal: np.ndarray,
    veto_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_volume = geometry.in_fiducial_area(loss_point)
    triggered = fiducial_signal.sum(axis=1) > 20
    vetoed = veto_signal.sum(axis=1) >= 1
    return in_volume, triggered, vetoed


def point_line_distances(xy, entry, exit):
    """
    :param xy: x,y positions to test
    :param entry: x,y position of entry point
    :param exit: x,y position of exit point
    """
    dx = exit[..., 0] - entry[..., 0]
    dy = exit[..., 1] - entry[..., 1]
    return (
        np.abs(
            dx[..., None] * (entry[..., 1, None] - xy[None, ..., 1])
            - dy[..., None] * (entry[..., 0, None] - xy[None, ..., 0])
        )
        / np.sqrt(dx**2 + dy**2)[..., None]
    )


def point_point_distances(xy, point):
    dx = xy[None, ..., 0] - point[..., 0, None]
    dy = xy[None, ..., 1] - point[..., 1, None]
    return np.sqrt(dx**2 + dy**2)
