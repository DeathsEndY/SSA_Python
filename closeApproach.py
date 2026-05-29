from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

RAD = np.pi / 180.0
GM_EARTH = 398600.4418  # km^3/s^2, 地球引力常数

@dataclass(frozen=True)
class MinAngle:
    con: float
    tar: float

@dataclass(frozen=True)
class ApproachConfig:
    base_epoch: Optional[Time] = None

@dataclass(frozen=True)
class ApproachInfo:
    time: Time
    con_angle: float
    tar_angle: float

@dataclass
class CElement:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    epoch: Time = field(default_factory=Time.now)
    _orbit: Orbit = field(init=False, repr=False)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)
        self._orbit = Orbit.from_vectors(
            Earth,
            self.pos * u.km,
            self.vel * u.km / u.s,
            epoch=self.epoch,
        )

    @classmethod
    def from_state(cls, epoch: Time, r: np.ndarray, v: np.ndarray) -> "CElement":
        return cls(pos=r, vel=v, epoch=epoch)

    @property
    def inclination_deg(self) -> float:
        return float(self._orbit.inc.to(u.deg).value)

    @property
    def raan_deg(self) -> float:
        return float(self._orbit.raan.to(u.deg).value)

    @property
    def raan_rad(self) -> float:
        return float(self._orbit.raan.to(u.rad).value)

    @property
    def argp_deg(self) -> float:
        return float(self._orbit.argp.to(u.deg).value)

    @property
    def argp_rad(self) -> float:
        return float(self._orbit.argp.to(u.rad).value)

    @property
    def ecc(self) -> float:
        return float(self._orbit.ecc.value)

    @property
    def true_anomaly_rad(self) -> float:
        return float(self._orbit.nu.to(u.rad).value)

    @property
    def true_anomaly_deg(self) -> float:
        return float(self._orbit.nu.to(u.deg).value)

    @property
    def semi_major_axis(self) -> float:
        return float(self._orbit.a.to(u.km).value)

    @property
    def period(self) -> float:
        return float(self._orbit.period.to(u.s).value)

    @property
    def mean_motion(self) -> float:
        return np.sqrt(GM_EARTH / self.semi_major_axis ** 3)

    @property
    def perigee_radius(self) -> float:
        return self.semi_major_axis * (1 - self.ecc)

    @property
    def apogee_radius(self) -> float:
        return self.semi_major_axis * (1 + self.ecc)

    def time_at_latitude(self, latitude_deg: float) -> Time:
        target_u = np.deg2rad(latitude_deg % 360.0)
        omega = self._orbit.argp.to(u.rad).value
        target_nu = (target_u - omega) % (2 * np.pi)

        if target_nu > np.pi:
            target_nu -= 2 * np.pi

        cosE = (self.ecc + np.cos(target_nu)) / (1 + self.ecc * np.cos(target_nu))
        cosE = np.clip(cosE, -1.0, 1.0)
        E = np.arccos(cosE)

        if target_nu < 0:
            E = 2 * np.pi - E

        m_target = E - self.ecc * np.sin(E)
        nu0 = self.true_anomaly_rad
        E0 = 2 * np.arctan(np.tan(nu0 / 2) * np.sqrt((1 - self.ecc) / (1 + self.ecc)))
        m0 = E0 - self.ecc * np.sin(E0)
        dM = (m_target - m0) % (2 * np.pi)
        dt = dM / self.mean_motion
        return self.epoch + TimeDelta(dt, format='sec')


def _orbit_normal_vector(inclination_deg: float, raan_deg: float) -> np.ndarray:
    return np.array([
        np.sin(inclination_deg * RAD) * np.sin(raan_deg * RAD),
        -np.sin(inclination_deg * RAD) * np.cos(raan_deg * RAD),
        np.cos(inclination_deg * RAD),
    ])


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def _intersecting_angles(tar_orb: CElement, con_orb: CElement, invert: bool = False) -> MinAngle:
    h_tar = _orbit_normal_vector(tar_orb.inclination_deg, tar_orb.raan_deg)
    h_con = _orbit_normal_vector(con_orb.inclination_deg, con_orb.raan_deg)
    w = np.cross(h_con, h_tar) if not invert else np.cross(h_tar, h_con)
    w = _normalize(w)

    if (w[2] < 0) if not invert else (w[2] > 0):
        w = -w

    nc = np.array([np.cos(con_orb.raan_rad), np.sin(con_orb.raan_rad), 0.0])
    nt = np.array([np.cos(tar_orb.raan_rad), np.sin(tar_orb.raan_rad), 0.0])
    con_angle = np.degrees(np.arccos(np.clip(np.dot(w, nc) / np.linalg.norm(nc), -1, 1)))
    tar_angle = np.degrees(np.arccos(np.clip(np.dot(w, nt) / np.linalg.norm(nt), -1, 1)))
    if invert:
        return MinAngle(con=360 - con_angle, tar=360 - tar_angle)
    return MinAngle(con=con_angle, tar=tar_angle)


def min_angle(tar_orb: CElement, con_orb: CElement) -> MinAngle:
    return _intersecting_angles(tar_orb, con_orb, invert=False)


def max_angle(tar_orb: CElement, con_orb: CElement) -> MinAngle:
    return _intersecting_angles(tar_orb, con_orb, invert=True)


def _base_epoch(tar_ini: CElement, con_ini: CElement, config: Optional[ApproachConfig] = None) -> Time:
    if config and config.base_epoch is not None:
        return config.base_epoch
    return tar_ini.epoch if (tar_ini.epoch - con_ini.epoch).sec > 0 else con_ini.epoch


def _find_approach_time(
    tar_sat,
    con_sat,
    num: int,
    angle_fn: Callable[[CElement, CElement], MinAngle],
    config: Optional[ApproachConfig] = None,
) -> ApproachInfo:
    tar_ini = tar_sat.initial_element()
    con_ini = con_sat.initial_element()
    start_time = _base_epoch(tar_ini, con_ini, config) + TimeDelta(num * tar_ini.period, format='sec')
    tar_iter = tar_sat.propagate(start_time)
    con_iter = con_sat.propagate(start_time)
    angles = angle_fn(tar_iter, con_iter)
    return ApproachInfo(
        time=tar_iter.time_at_latitude(angles.tar),
        con_angle=angles.con,
        tar_angle=angles.tar,
    )


def find_arg_time(
    tar_sat,
    con_sat,
    num: int,
    config: Optional[ApproachConfig] = None,
) -> ApproachInfo:
    return _find_approach_time(tar_sat, con_sat, num, min_angle, config)


def find_dec_time(
    tar_sat,
    con_sat,
    num: int,
    config: Optional[ApproachConfig] = None,
) -> ApproachInfo:
    return _find_approach_time(tar_sat, con_sat, num, max_angle, config)


def cubic_splining(p1, p2, p3, p4, t1, t2):
    c0 = p1
    det = t1**3 * t2**2 + t1**2 * t2 + t1 * t2**3 - t1**3 * t2 - t1**2 * t2**3 - t1 * t2**2
    c1 = (
        (t2**3 - t2**2) * (p2 - p1)
        + (t1**2 - t1**3) * (p3 - p1)
        + (t1**3 * t2**2 - t1**2 * t2**3) * (p4 - p1)
    ) / det
    c2 = (
        (t2 - t2**3) * (p2 - p1)
        + (t1**3 - t1) * (p3 - p1)
        + (t1 * t2**3 - t1**3 * t2) * (p4 - p1)
    ) / det
    c3 = (
        (t2**2 - t2) * (p2 - p1)
        + (t1 - t1**2) * (p3 - p1)
        + (t1**2 * t2 - t1 * t2**2) * (p4 - p1)
    ) / det
    p = c2 / c3
    q = c1 / c3
    r = c0 / c3
    a = (3 * q - p * p) / 3.0
    b = (2 * p**3 - 9 * p * q + 27 * r) / 27.0
    delta = a**3 / 27.0 + b**2 / 4.0

    if delta > 0:
        root1 = -b / 2.0 + np.sqrt(delta)
        root2 = -b / 2.0 - np.sqrt(delta)
        return np.cbrt(root1) + np.cbrt(root2)

    e0 = 2 * np.sqrt(-a / 3.0)
    cosphi = -b / 2.0 / np.sqrt(-a**3 / 27.0)
    cosphi = np.clip(cosphi, -1, 1)
    phi = np.arccos(cosphi)
    z3 = e0 * np.cos(phi / 3.0 + 4 * np.pi / 3.0)
    return z3 - p / 3.0
    
