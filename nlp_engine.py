from __future__ import annotations

import re
import os
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import dateparser
import nltk
from nltk.stem import PorterStemmer

# Optional sentiment (VADER). If not available, we ignore sentiment safely.
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
except Exception:
    _SIA = None

stemmer = PorterStemmer()

# Stopwords are optional. If missing, we use a minimal list.
_MIN_STOP = {"a","an","the","is","are","was","were","in","on","at","to","for","of","and","or","please","tell","me","what","whats","what's","give","show","can","could","will"}


WEATHER_INTENTS = [
    "temperature_now",
    "conditions_now",
    "wind_now",
    "humidity_now",
    "rain_now",
    "snow_now",
    "tomorrow_summary",
    "tomorrow_rain",
    "next3days_summary",
    "week_summary",
    "help",
]

MODEL_PATH = os.path.join("models", "intent_model.pkl")
TRAINING_PATH = os.path.join("data", "training_intents.json")


def _tokenize(text: str):
    # Regex tokenizer: no NLTK "punkt" download needed
    return re.findall(r"[a-zA-Z']+", text.lower())


def _preprocess(text: str):
    toks = [t for t in _tokenize(text) if t not in _MIN_STOP]
    stems = [stemmer.stem(t) for t in toks]
    return stems


def _features_from_tokens(tokens):
    return {t: True for t in tokens}


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


_MODEL = _load_model()


def _predict_intent_ml(text: str) -> Optional[str]:
    if _MODEL is None:
        return None
    tokens = _preprocess(text)
    feats = _features_from_tokens(tokens)
    try:
        return _MODEL.classify(feats)
    except Exception:
        return None


def _predict_intent_rules(text: str) -> str:
    t = text.lower()

    # Help
    if any(x in t for x in ["help", "what can you do", "commands", "examples"]):
        return "help"

    # Time horizons
    has_tomorrow = "tomorrow" in t
    has_week = any(x in t for x in ["this week", "next 7", "7 day", "week forecast", "weekly"])
    has_3day = any(x in t for x in ["next 3", "3 day", "three day", "next three"])

    # Variables
    if any(x in t for x in ["temperature", "temp", "how hot", "how cold", "degrees"]):
        return "temperature_now"
    if any(x in t for x in ["wind", "windy", "gust"]):
        return "wind_now"
    if any(x in t for x in ["humidity", "humid"]):
        return "humidity_now"
    if any(x in t for x in ["snow", "snowfall"]):
        return "snow_now" if not has_tomorrow else "tomorrow_summary"
    if any(x in t for x in ["rain", "raining", "precip", "shower"]):
        if has_tomorrow:
            return "tomorrow_rain"
        return "rain_now"
    if any(x in t for x in ["condition", "weather like", "sunny", "cloudy", "clear", "overcast"]):
        return "conditions_now"

    # Default by time horizon
    if has_tomorrow:
        return "tomorrow_summary"
    if has_3day:
        return "next3days_summary"
    if has_week:
        return "week_summary"

    return "conditions_now"


def _extract_location(text: str) -> Optional[str]:
    # common patterns: "in Berlin", "at New York", "for Tokyo"
    m = re.search(r"\b(?:in|at|for)\s+([a-zA-Z\s,.'-]{2,})$", text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # "weather Berlin"
    m = re.search(r"\bweather\s+in\s+([a-zA-Z\s,.'-]{2,})", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # If message ends with a capitalized place-like token sequence
    m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s*,\s*[A-Z][a-z]+)?)\s*$", text)
    if m and len(text.split()) >= 2:
        guess = m.group(1).strip()
        # avoid matching "Tomorrow" etc.
        if guess.lower() not in {"tomorrow", "today", "week", "morning", "evening"}:
            return guess
    return None


def _extract_time_window(text: str):
    t = text.lower()
    # Default: current
    horizon = "now"
    if "tomorrow" in t:
        horizon = "tomorrow"
    elif any(x in t for x in ["next 3", "3 day", "three day"]):
        horizon = "next3days"
    elif any(x in t for x in ["week", "7 day", "next 7"]):
        horizon = "week"

    # Time-of-day buckets (optional)
    tod = None
    if "morning" in t:
        tod = "morning"
    elif "afternoon" in t:
        tod = "afternoon"
    elif "evening" in t:
        tod = "evening"
    elif "night" in t:
        tod = "night"

    return horizon, tod


def _sentiment(text: str) -> Optional[str]:
    if _SIA is None:
        return None
    s = _SIA.polarity_scores(text).get("compound", 0.0)
    if s >= 0.35:
        return "positive"
    if s <= -0.35:
        return "negative"
    return "neutral"


def interpret_message(message: str, last_location: Optional[str], lat: Any, lon: Any) -> Dict[str, Any]:
    msg = (message or "").strip()

    # coords provided via browser geolocation
    coords = None
    if lat is not None and lon is not None:
        try:
            coords = {"lat": float(lat), "lon": float(lon)}
        except Exception:
            coords = None

    location = _extract_location(msg) or last_location

    horizon, tod = _extract_time_window(msg)

    # intent: ML first, then rules
    intent = _predict_intent_ml(msg) or _predict_intent_rules(msg)

    if not coords and not location:
        return {
            "intent": intent,
            "horizon": horizon,
            "tod": tod,
            "coords": None,
            "location_label": None,
            "sentiment": _sentiment(msg),
            "error": "Which location should I use? Example: **"Will it rain in Lisbon tomorrow morning?"**",
        }

    return {
        "intent": intent,
        "horizon": horizon,
        "tod": tod,
        "coords": coords,
        "location_label": location,
        "sentiment": _sentiment(msg),
        "error": None,
    }


# -------- Response formatting --------

_WEATHER_CODE = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle (light)",
    57: "freezing drizzle (dense)",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain (light)",
    67: "freezing rain (heavy)",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "rain showers (slight)",
    81: "rain showers (moderate)",
    82: "rain showers (violent)",
    85: "snow showers (slight)",
    86: "snow showers (heavy)",
    95: "thunderstorm",
    96: "thunderstorm with hail (slight)",
    99: "thunderstorm with hail (heavy)",
}


def _tod_hours(tod: Optional[str]):
    if tod == "morning":
        return (6, 12)
    if tod == "afternoon":
        return (12, 18)
    if tod == "evening":
        return (18, 24)
    if tod == "night":
        return (0, 6)
    return None


def format_weather_reply(parsed: Dict[str, Any], bundle: Dict[str, Any], location_label: str) -> str:
    intent = parsed["intent"]
    horizon = parsed["horizon"]
    tod = parsed.get("tod")

    cur = bundle.get("current", {})
    daily = bundle.get("daily", [])
    hourly = bundle.get("hourly", [])

    # Slightly human tone if negative sentiment (optional)
    sent = parsed.get("sentiment")
    tone_prefix = ""
    if sent == "negative":
        tone_prefix = "I’ve got you — "

    # Now / current
    if intent in {"temperature_now", "conditions_now", "wind_now", "humidity_now", "rain_now", "snow_now"} and horizon == "now":
        parts = [f"**{location_label}** right now:"]
        if intent in {"temperature_now", "conditions_now"}:
            temp = cur.get("temperature_c")
            code = cur.get("weather_code")
            desc = _WEATHER_CODE.get(code, "unknown")
            if temp is not None:
                parts.append(f"- Temperature: **{temp:.1f}°C**")
            parts.append(f"- Conditions: **{desc}**")
        if intent == "wind_now":
            w = cur.get("wind_kmh")
            if w is not None:
                parts.append(f"- Wind: **{w:.0f} km/h**")
        if intent == "humidity_now":
            h = cur.get("humidity_pct")
            if h is not None:
                parts.append(f"- Humidity: **{h:.0f}%**")
        if intent == "rain_now":
            r = cur.get("rain_mm")
            if r is not None:
                parts.append(f"- Rain (last hour): **{r:.1f} mm**")
        if intent == "snow_now":
            s = cur.get("snow_mm")
            if s is not None:
                parts.append(f"- Snowfall (last hour): **{s:.1f} mm**")
        return tone_prefix + "\n".join(parts)

    # Tomorrow
    if horizon == "tomorrow":
        if not daily or len(daily) < 2:
            return "I couldn't get tomorrow's forecast right now. Please try again."
        d = daily[1]
        date = d.get("date")
        desc = _WEATHER_CODE.get(d.get("weather_code"), "unknown")
        tmin = d.get("tmin_c")
        tmax = d.get("tmax_c")
        rain = d.get("rain_mm")
        snow = d.get("snow_mm")
        pop = d.get("precip_prob_max")

        # If asked about rain tomorrow specifically:
        if intent == "tomorrow_rain":
            msg = f"**{location_label}** tomorrow ({date}): "
            if pop is not None:
                msg += f"precipitation probability up to **{pop:.0f}%**. "
            if rain is not None:
                msg += f"expected rain ≈ **{rain:.1f} mm**."
            return tone_prefix + msg

        # If time of day requested, use hourly window
        hours = _tod_hours(tod)
        if hours and hourly:
            start, end = hours
            # Filter tomorrow hours in local time string "YYYY-MM-DDTHH:MM"
            vals = [h for h in hourly if h["date_offset"] == 1 and start <= h["hour"] < end]
            if vals:
                pop_avg = sum(v.get("precip_prob", 0) for v in vals) / len(vals)
                rain_sum = sum(v.get("rain_mm", 0) for v in vals)
                return tone_prefix + (f"**{location_label}** tomorrow {tod}: "
                                      f"avg precip prob ~ **{pop_avg:.0f}%**, rain sum ~ **{rain_sum:.1f} mm**.")
        return tone_prefix + (
            f"**{location_label}** tomorrow ({date}): **{desc}**\n"
            f"- Min/Max: **{tmin:.1f}°C / {tmax:.1f}°C**\n"
            f"- Rain: **{rain:.1f} mm**, Snow: **{snow:.1f} mm**\n"
            f"- Precip. probability (max): **{pop:.0f}%**"
        )

    # Next 3 days
    if horizon == "next3days":
        if len(daily) < 3:
            return "I couldn't get the next 3 days right now. Please try again."
        lines = [f"**{location_label}** – next 3 days:"]
        for i in range(0, 3):
            d = daily[i]
            desc = _WEATHER_CODE.get(d.get("weather_code"), "unknown")
            lines.append(f"- {d['date']}: {desc}, {d['tmin_c']:.1f}–{d['tmax_c']:.1f}°C, rain {d['rain_mm']:.1f} mm")
        return tone_prefix + "\n".join(lines)

    # Week
    if horizon == "week":
        if len(daily) < 7:
            return "I couldn't get the weekly forecast right now. Please try again."
        lines = [f"**{location_label}** – 7-day outlook:"]
        for i in range(0, 7):
            d = daily[i]
            desc = _WEATHER_CODE.get(d.get("weather_code"), "unknown")
            lines.append(f"- {d['date']}: {desc}, {d['tmin_c']:.1f}–{d['tmax_c']:.1f}°C")
        return tone_prefix + "\n".join(lines)

    # Help
    if intent == "help":
        return (
            "You can ask about:\n"
            "- temperature, conditions (sunny/cloudy), wind, humidity, rain/snow\n"
            "- tomorrow’s forecast, next 3 days, or weekly outlook\n\n"
            "Examples:\n"
            "• *What's the wind speed in Tokyo?*\n"
            "• *Will it rain in Lisbon tomorrow morning?*\n"
            "• *3-day forecast for New York*"
        )

    # Default fallback
    return (
        "I can help with weather like temperature, conditions, wind, humidity, rain/snow, "
        "and forecasts (tomorrow / 3-day / week). Try: **'weather in Berlin'**."
    )
