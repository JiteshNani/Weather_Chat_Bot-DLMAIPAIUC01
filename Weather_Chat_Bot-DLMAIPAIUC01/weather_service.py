from __future__ import annotations

import time
import requests
from typing import Dict, Optional, Tuple, Any

# Open-Meteo base URLs (no API key required)
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Simple in-memory cache
_CACHE = {}
_TTL_SECONDS = 300  # 5 minutes


def _cache_get(key: str):
    v = _CACHE.get(key)
    if not v:
        return None
    ts, data = v
    if time.time() - ts > _TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return data


def _cache_set(key: str, data: Any):
    _CACHE[key] = (time.time(), data)


def geocode_location(query: str) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    key = f"geo:{query.lower().strip()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        r = requests.get(GEOCODE_URL, params={"name": query, "count": 1, "language": "en", "format": "json"}, timeout=10)
        r.raise_for_status()
        js = r.json()
        results = js.get("results") or []
        if not results:
            _cache_set(key, None)
            return None
        top = results[0]
        data = {
            "name": f"{top.get('name')}, {top.get('country')}",
            "latitude": float(top["latitude"]),
            "longitude": float(top["longitude"]),
        }
        _cache_set(key, data)
        return data
    except Exception:
        return None


def get_forecast_bundle(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    key = f"wx:{lat:.4f},{lon:.4f}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,weather_code",
        "hourly": "temperature_2m,precipitation_probability,rain,snowfall,relative_humidity_2m,wind_speed_10m,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,rain_sum,snowfall_sum",
        "forecast_days": 7,
    }

    try:
        r = requests.get(FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()

        bundle = _parse_bundle(js)
        _cache_set(key, bundle)
        return bundle
    except Exception:
        return None


def _parse_bundle(js: Dict[str, Any]) -> Dict[str, Any]:
    cur = js.get("current") or {}
    daily = js.get("daily") or {}
    hourly = js.get("hourly") or {}

    # Current
    current = {
        "temperature_c": cur.get("temperature_2m"),
        "apparent_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "precip_mm": cur.get("precipitation"),
        "rain_mm": cur.get("rain"),
        "snow_mm": cur.get("snowfall"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "weather_code": cur.get("weather_code"),
        "time": cur.get("time"),
    }

    # Daily list
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    wcode = daily.get("weather_code") or []
    pop = daily.get("precipitation_probability_max") or []
    rain = daily.get("rain_sum") or []
    snow = daily.get("snowfall_sum") or []

    daily_list = []
    for i in range(min(len(dates), 7)):
        daily_list.append({
            "date": dates[i],
            "tmax_c": float(tmax[i]) if i < len(tmax) and tmax[i] is not None else None,
            "tmin_c": float(tmin[i]) if i < len(tmin) and tmin[i] is not None else None,
            "weather_code": int(wcode[i]) if i < len(wcode) and wcode[i] is not None else None,
            "precip_prob_max": float(pop[i]) if i < len(pop) and pop[i] is not None else None,
            "rain_mm": float(rain[i]) if i < len(rain) and rain[i] is not None else 0.0,
            "snow_mm": float(snow[i]) if i < len(snow) and snow[i] is not None else 0.0,
        })

    # Hourly list (we also store offset from "now" day index to help tomorrow windows)
    ht = hourly.get("time") or []
    hpop = hourly.get("precipitation_probability") or []
    hrain = hourly.get("rain") or []
    hsnow = hourly.get("snowfall") or []
    hcode = hourly.get("weather_code") or []
    hwind = hourly.get("wind_speed_10m") or []
    hhum = hourly.get("relative_humidity_2m") or []

    # Determine "today" date prefix
    now_iso = current.get("time") or (ht[0] if ht else "")
    today = now_iso.split("T")[0] if "T" in now_iso else (now_iso[:10] if now_iso else "")

    hourly_list = []
    for i in range(len(ht)):
        iso = ht[i]
        date = iso.split("T")[0] if "T" in iso else iso[:10]
        hour = int(iso.split("T")[1].split(":")[0]) if "T" in iso else 0
        # date offset relative to today (0 today, 1 tomorrow, etc.)
        offset = 0
        if today and date:
            try:
                from datetime import date as _d
                y1,m1,d1 = map(int, today.split("-"))
                y2,m2,d2 = map(int, date.split("-"))
                offset = (_d(y2,m2,d2) - _d(y1,m1,d1)).days
            except Exception:
                offset = 0

        hourly_list.append({
            "time": iso,
            "date_offset": offset,
            "hour": hour,
            "precip_prob": float(hpop[i]) if i < len(hpop) and hpop[i] is not None else 0.0,
            "rain_mm": float(hrain[i]) if i < len(hrain) and hrain[i] is not None else 0.0,
            "snow_mm": float(hsnow[i]) if i < len(hsnow) and hsnow[i] is not None else 0.0,
            "weather_code": int(hcode[i]) if i < len(hcode) and hcode[i] is not None else None,
            "wind_kmh": float(hwind[i]) if i < len(hwind) and hwind[i] is not None else None,
            "humidity_pct": float(hhum[i]) if i < len(hhum) and hhum[i] is not None else None,
        })

    return {"current": current, "daily": daily_list, "hourly": hourly_list}
