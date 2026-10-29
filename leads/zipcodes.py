"""ZIP -> (city, state) resolution.

Discovery quality depends on knowing the city behind a ZIP code: search queries
are city-based, not ZIP-based. This module indexes whichever US ZIP dataset is
present, with a prefix-table fallback so the system still runs if the CSV files
are missing.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# USPS ZIP prefix -> state. Used only when no ZIP dataset is available.
ZIP_PREFIX_RANGES: tuple[tuple[str, int, int], ...] = (
    ("AK", 995, 999), ("AL", 350, 369), ("AR", 716, 729), ("AZ", 850, 865),
    ("CA", 900, 961), ("CO", 800, 816), ("CT", 60, 69), ("DC", 200, 205),
    ("DE", 197, 199), ("FL", 320, 349), ("GA", 300, 319), ("HI", 967, 968),
    ("IA", 500, 528), ("ID", 832, 838), ("IL", 600, 629), ("IN", 460, 479),
    ("KS", 660, 679), ("KY", 400, 427), ("LA", 700, 714), ("MA", 10, 27),
    ("MD", 210, 219), ("ME", 39, 49), ("MI", 480, 499), ("MN", 550, 567),
    ("MO", 630, 658), ("MS", 386, 397), ("MT", 590, 599), ("NC", 270, 289),
    ("ND", 580, 588), ("NE", 680, 693), ("NH", 30, 38), ("NJ", 70, 89),
    ("NM", 870, 884), ("NV", 889, 898), ("NY", 100, 149), ("OH", 430, 459),
    ("OK", 730, 749), ("OR", 970, 979), ("PA", 150, 196), ("RI", 28, 29),
    ("SC", 290, 299), ("SD", 570, 577), ("TN", 370, 385), ("TX", 750, 799),
    ("UT", 840, 847), ("VA", 220, 246), ("VT", 50, 59), ("WA", 980, 994),
    ("WI", 530, 549), ("WV", 247, 268), ("WY", 820, 831),
)


def state_from_zip(zip_code: str) -> str:
    try:
        prefix = int(str(zip_code).zfill(5)[:3])
    except (TypeError, ValueError):
        return "UNKNOWN"
    for state, low, high in ZIP_PREFIX_RANGES:
        if low <= prefix <= high:
            return state
    return "UNKNOWN"


@dataclass(frozen=True)
class ZipLocation:
    zip: str
    city: str
    state: str
    lat: float | None = None
    lon: float | None = None


def _pick(headers: list[str], *candidates: str) -> str | None:
    lowered = {h.strip().lower().lstrip("\ufeff"): h for h in headers}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


class ZipIndex:
    """Loads a US ZIP dataset once and answers ``lookup(zip)``."""

    def __init__(self, data: dict[str, ZipLocation]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def from_file(cls, path: Path | str) -> "ZipIndex":
        path = Path(path)
        if not path.exists():
            return cls({})
        data: dict[str, ZipLocation] = {}
        try:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                headers = reader.fieldnames or []
                zip_col = _pick(headers, "zip", "postal code", "postal_code", "zcta")
                city_col = _pick(headers, "city", "place name", "place_name")
                state_col = _pick(headers, "state_id", "admin code1", "state")
                lat_col = _pick(headers, "lat", "latitude")
                lon_col = _pick(headers, "lng", "longitude", "lon", "long")
                if not zip_col:
                    log.warning("ZIP dataset %s has no zip column", path)
                    return cls({})

                for row in reader:
                    z = (row.get(zip_col) or "").strip().zfill(5)
                    if len(z) != 5 or not z.isdigit():
                        continue
                    state = (row.get(state_col) or "").strip().upper() if state_col else ""
                    if len(state) != 2:
                        state = state_from_zip(z)
                    lat = lon = None
                    try:
                        lat = float(row[lat_col]) if lat_col and row.get(lat_col) else None
                        lon = float(row[lon_col]) if lon_col and row.get(lon_col) else None
                    except (TypeError, ValueError):
                        pass
                    data[z] = ZipLocation(
                        zip=z,
                        city=(row.get(city_col) or "").strip() if city_col else "",
                        state=state,
                        lat=lat,
                        lon=lon,
                    )
        except (OSError, csv.Error) as exc:
            log.warning("could not read ZIP dataset %s: %s", path, exc)
            return cls({})
        log.debug("loaded %d ZIP codes from %s", len(data), path)
        return cls(data)

    @classmethod
    def from_candidates(cls, paths: list[Path | str]) -> "ZipIndex":
        for p in paths:
            idx = cls.from_file(p)
            if len(idx) > 1000:
                return idx
        return cls({})

    def lookup(self, zip_code: str) -> ZipLocation:
        z = str(zip_code).zfill(5)
        found = self._data.get(z)
        if found:
            return found
        return ZipLocation(zip=z, city="", state=state_from_zip(z))
