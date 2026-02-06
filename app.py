from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from nlp_engine import interpret_message, format_weather_reply
from weather_service import (
    geocode_location,
    get_forecast_bundle,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or "").strip()
    lat = payload.get("lat")
    lon = payload.get("lon")

    if not msg and (lat is None or lon is None):
        return jsonify({"reply": "Please type a weather question or click **Use my location**."})

    # Interpret user request (intent + slots)
    parsed = interpret_message(
        msg,
        last_location=session.get("last_location"),
        lat=lat,
        lon=lon,
    )

    if parsed["error"]:
        return jsonify({"reply": parsed["error"]})

    # Resolve coordinates
    if parsed["coords"]:
        coords = parsed["coords"]
        location_label = parsed.get("location_label") or "your location"
        session["last_location"] = location_label
    else:
        geo = geocode_location(parsed["location_label"])
        if geo is None:
            return jsonify({"reply": f"I couldn't find **{parsed['location_label']}**. Try a city + country (e.g., *Lisbon, Portugal*)."})
        coords = {"lat": geo["latitude"], "lon": geo["longitude"]}
        location_label = geo["name"]
        session["last_location"] = location_label

    # Fetch weather
    bundle = get_forecast_bundle(coords["lat"], coords["lon"])
    if bundle is None:
        return jsonify({"reply": "Weather service is temporarily unavailable. Please try again."})

    reply = format_weather_reply(parsed, bundle, location_label)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
