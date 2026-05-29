from __future__ import annotations

import numpy as np
from sgp4.api import Satrec, WGS72
from astropy.time import Time, TimeDelta
from typing import Union
from closeApproach import CElement

class CSatellite:
    def __init__(self, line1: str, line2: str, line3: str):
        self.line1 = line1.strip()
        self.line2 = line2.strip()
        self.line3 = line3.strip()
        self.satrec = Satrec.twoline2rv(self.line2, self.line3, WGS72)
        self._epoch = self._tle_to_epoch()

    @property
    def name(self) -> str:
        return self.line1[2:].strip()

    @property
    def sat_id(self) -> str:
        return self.line2[2:7].strip()

    @property
    def epoch(self) -> Time:
        return self._epoch

    def _tle_to_epoch(self) -> Time:
        year = int(self.line2[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(self.line2[20:32])
        return Time(f"{year}-01-01T00:00:00", scale="utc") + TimeDelta(day_of_year - 1, format="jd")

    def _state_at_jd(self, jd: float):
        error_code, position, velocity = self.satrec.sgp4(jd, 0.0)
        if error_code != 0:
            raise RuntimeError(f"sgp4 error {error_code}")
        return np.array(position), np.array(velocity)

    def initial_element(self) -> CElement:
        position, velocity = self._state_at_jd(self._epoch.jd)
        return CElement.from_state(self._epoch, position, velocity)

    def propagate(self, target_time: Union[Time, int, float]) -> CElement:
        if isinstance(target_time, (int, float)):
            target_time = self._epoch + TimeDelta(float(target_time), format="sec")
        elif not isinstance(target_time, Time):
            raise TypeError("target_time must be astropy.time.Time or a number of seconds")

        position, velocity = self._state_at_jd(target_time.jd)
        return CElement.from_state(target_time, position, velocity)