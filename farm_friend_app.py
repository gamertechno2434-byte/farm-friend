"""Farm Friend backend (Python).

One file, next to index.html. Serves the app and adds:
  - NASA AppEEARS satellite analysis  (MOD13Q1.061, NDVI + EVI, 250 m, 16 days)
  - Open-Meteo weather
  - OpenAI plain-language advice, chat and plant-photo check

Run:      python farm_friend_app.py
Open:     http://127.0.0.1:8000      (the app opens by itself)

Optional .env file next to this script:
  OPENAI_API_KEY=sk-...
  or GEMINI_API_KEY=...              free key from https://aistudio.google.com/app/apikey
  EARTHDATA_USERNAME=your_nasa_login
  EARTHDATA_PASSWORD=your_nasa_password
  OPENAI_MODEL=gpt-4.1               (optional)
  GEMINI_MODEL=gemini-2.5-flash      (optional)
  OPENAI_BASE_URL=...                (optional OpenAI-compatible endpoint)
  PORT=8000                          (optional)

Install:  pip install -r requirements.txt
"""
import os
import re
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

NASA_API = "https://appeears.earthdatacloud.nasa.gov/api/"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
OPENAI_API = "https://api.openai.com/v1/chat/completions"
PRODUCT = "MOD13Q1.061"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CROP_HEAT = {
    "Wheat": 32, "Tomato": 35, "Maize": 35, "Rice": 35, "Date palm": 45,
    "Cucumber": 34, "Vegetables": 34, "Potato": 28, "Cotton": 38, "Barley": 30,
}
VERSION = "5.0.0"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MAX_IMAGE_CHARS = 12_000_000  # about 9 MB of image data


def ai_settings():
    """OpenAI by default; a free Google AI Studio key can be used instead."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if openai_key:
        return (
            openai_key,
            os.getenv("OPENAI_MODEL", "").strip() or OPENAI_MODEL,
            "openai",
        )
    if gemini_key:
        return (
            gemini_key,
            os.getenv("GEMINI_MODEL", "").strip() or GEMINI_MODEL,
            "gemini",
        )
    return None, None, None


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def summarize_weather(weather, crop):
    """Turn the Open-Meteo payload into a few numbers a model cannot misread."""
    if not isinstance(weather, dict):
        return {"available": False}
    now = weather.get("current") or {}
    daily = weather.get("daily") or {}
    dates = daily.get("time") or []

    def series(name):
        return [_num(v) for v in (daily.get(name) or [])[:7]]

    rain, tmax, tmin = series("precipitation_sum"), series("temperature_2m_max"), series("temperature_2m_min")
    gust, et0 = series("wind_gusts_10m_max"), series("et0_fao_evapotranspiration")
    days = []
    for i, day in enumerate(dates[:7]):
        days.append({
            "date": day,
            "rain_mm": rain[i] if i < len(rain) else None,
            "tmax_c": tmax[i] if i < len(tmax) else None,
            "tmin_c": tmin[i] if i < len(tmin) else None,
            "gust_kmh": gust[i] if i < len(gust) else None,
            "et0_mm": et0[i] if i < len(et0) else None,
        })
    rain_vals = [v for v in rain if v is not None]
    tmax_vals = [v for v in tmax if v is not None]
    gust_vals = [v for v in gust if v is not None]
    et0_vals = [v for v in et0 if v is not None]
    heat_limit = CROP_HEAT.get(crop, 35)
    return {
        "now": {
            "temp_c": _num(now.get("temperature_2m")),
            "humidity_pct": _num(now.get("relative_humidity_2m")),
            "rain_mm": _num(now.get("precipitation")),
            "wind_kmh": _num(now.get("wind_speed_10m")),
        },
        "days": days,
        "rain_next_7d_mm": round(sum(rain_vals), 1) if rain_vals else None,
        "dry_days_next_7d": sum(1 for v in rain if v is not None and v < 1),
        "crop_heat_limit_c": heat_limit,
        "days_at_or_above_heat_limit": sum(1 for v in tmax_vals if v >= heat_limit),
        "max_gust_kmh": round(max(gust_vals)) if gust_vals else None,
        "et0_next_7d_mm": round(sum(et0_vals), 1) if et0_vals else None,
    }


def summarize_satellite(data):
    """Condense the NDVI/EVI record so the model gets trend, not just rows."""
    if data is None or data.empty:
        return {"readings": 0}
    nd = data.dropna(subset=["_ndvi"]).sort_values("_date")
    ev = data.dropna(subset=["_evi"]).sort_values("_date")
    if nd.empty and ev.empty:
        return {"readings": 0}
    out = {
        "product": "MOD13Q1.061",
        "pixel_m": 250,
        "composite_days": 16,
        "readings": int(len(nd) + len(ev)),
    }
    if not nd.empty:
        first, last = float(nd.iloc[0]["_ndvi"]), float(nd.iloc[-1]["_ndvi"])
        change = last - first
        out.update({
            "latest_date": str(nd.iloc[-1]["_date"].date()),
            "ndvi_latest": round(last, 3),
            "ndvi_change": round(change, 3),
            "trend": "falling" if change <= -0.05 else "rising" if change >= 0.05 else "steady",
            "ndvi_min": round(float(nd["_ndvi"].min()), 3),
            "ndvi_max": round(float(nd["_ndvi"].max()), 3),
        })
    if not ev.empty:
        out["evi_latest"] = round(float(ev.iloc[-1]["_evi"]), 3)
        out["evi_latest_date"] = str(ev.iloc[-1]["_date"].date())
    return out


def tidy_ai(text):
    """Models still add markdown; the page and the voice reader show it literally."""
    text = (text or "").strip()
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:1600]


def ai_reply(system, user_text, image=None, max_tokens=450):
    """One call to OpenAI or Gemini. Returns plain text, or "" without a key.

    New AI Studio keys start with "AQ." and are rejected by Gemini's
    OpenAI-compatible endpoint, so Gemini uses its native generateContent API.
    """
    key, model, provider = ai_settings()
    if not key:
        return ""
    if provider == "gemini":
        parts = [{"text": user_text}]
        if image:
            header, data = image.split(",", 1)
            parts.append({
                "inline_data": {
                    "mime_type": header.split(";")[0].replace("data:", ""),
                    "data": data,
                }
            })
        r = requests.post(
            f"{GEMINI_API}/{model}:generateContent",
            timeout=90,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        if not r.ok:
            try:
                detail = r.json()["error"]["message"]
            except (KeyError, TypeError, ValueError):
                detail = r.text[:300]
            raise RuntimeError(f"Gemini error {r.status_code}: {detail}")
        try:
            text = "".join(
                part.get("text", "")
                for part in r.json()["candidates"][0]["content"]["parts"]
            )
        except (KeyError, IndexError, TypeError):
            text = ""
    else:
        base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        content = user_text
        if image:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        r = requests.post(
            (base + "/chat/completions") if base else OPENAI_API,
            timeout=90,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
        )
        if not r.ok:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")
        try:
            text = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = ""
    return tidy_ai(text) if text else ""


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"
LEGACY_HTML = BASE_DIR / "farm_friend.html"
DATA_DIR = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
jobs, jobs_lock = {}, threading.Lock()

app = FastAPI(title="Farm Friend", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=str(DATA_DIR)), name="results")


# ---------------------------------------------------------------- models
class FarmRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    crop: str = Field(min_length=1, max_length=100)
    growth_stage: str = "Unknown"
    irrigation: str = "Unknown"
    soil: str = "Unknown"
    start_date: str
    end_date: str
    earthdata_username: str = ""
    earthdata_password: str = ""


class PhotoRequest(BaseModel):
    image: str = Field(min_length=20)
    crop: str = ""
    stage: str = ""
    question: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    crop: str = ""
    stage: str = ""
    water: str = ""
    soil: str = ""
    weather: dict | None = None
    satellite: dict | None = None


# ---------------------------------------------------------------- jobs
def update_job(job_id, **values):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)


def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


# ---------------------------------------------------------------- NASA
def api_get(session, url):
    r = session.get(url, timeout=45)
    r.raise_for_status()
    return r.json()


def authenticate(session, user, pw):
    r = session.post(
        NASA_API + "login",
        auth=(user, pw),
        data={"grant_type": "client_credentials"},
        timeout=45,
    )
    r.raise_for_status()
    token = r.json().get("token")
    if not token:
        raise RuntimeError("NASA did not return a login token.")
    session.headers.update({"Authorization": f"Bearer {token}"})


def discover_layers(session):
    names = list(api_get(session, f"{NASA_API}product/{PRODUCT}").keys())
    ndvi = next((n for n in names if n.lower() == "_250m_16_days_ndvi"), None)
    evi = next((n for n in names if n.lower() == "_250m_16_days_evi"), None)
    if not ndvi or not evi:
        raise RuntimeError("Could not find the MODIS NDVI and EVI layers.")
    return ndvi, evi


def appeears_date(value):
    """Our API uses DD-MM-YYYY; AppEEARS expects MM-DD-YYYY."""
    return datetime.strptime(value, "%d-%m-%Y").strftime("%m-%d-%Y")


def create_task(session, farm):
    ndvi, evi = discover_layers(session)
    task = {
        "task_type": "point",
        "task_name": "farm_friend_" + uuid.uuid4().hex[:8],
        "params": {
            "dates": [
                {
                    "startDate": appeears_date(farm.start_date),
                    "endDate": appeears_date(farm.end_date),
                }
            ],
            "layers": [
                {"product": PRODUCT, "layer": ndvi},
                {"product": PRODUCT, "layer": evi},
            ],
            "coordinates": [
                {
                    "id": "field",
                    "latitude": farm.latitude,
                    "longitude": farm.longitude,
                }
            ],
        },
    }
    r = session.post(NASA_API + "task", json=task, timeout=60)
    r.raise_for_status()
    task_id = r.json().get("task_id")
    if not task_id:
        raise RuntimeError("NASA did not return a task ID.")
    return task_id


def wait_for_task(session, task_id, job_id):
    started = time.monotonic()
    while True:
        status = str(
            api_get(session, f"{NASA_API}task/{task_id}").get("status", "unknown")
        ).lower()
        update_job(
            job_id,
            nasa_status=status,
            message=f"NASA is working on it (status: {status}). This can take 5 to 30 minutes.",
        )
        if status == "done":
            return
        if status in {"error", "failed", "cancelled"}:
            raise RuntimeError(f"NASA task ended with status: {status}")
        if time.monotonic() - started > 3600:
            raise TimeoutError("NASA processing timed out.")
        time.sleep(15)


def download_files(session, task_id, job_dir):
    files = api_get(session, f"{NASA_API}bundle/{task_id}").get("files", [])
    csvs = []
    for item in files:
        file_id = item.get("file_id")
        if not file_id:
            continue
        r = session.get(f"{NASA_API}bundle/{task_id}/{file_id}", timeout=180)
        r.raise_for_status()
        dest = job_dir / Path(item.get("file_name", "output")).name
        dest.write_bytes(r.content)
        if dest.suffix.lower() == ".csv":
            csvs.append(dest)
    if not csvs:
        raise RuntimeError("NASA returned no CSV files for this request.")
    return csvs


def find_column(columns, keyword):
    keyword = keyword.lower()
    for col in columns:
        if str(col).strip().lower() == keyword:
            return col
    for col in columns:
        if keyword in str(col).lower():
            return col
    return None


def load_data(csv_files):
    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        if df.empty:
            continue
        ndvi_col = find_column(df.columns, "NDVI")
        evi_col = find_column(df.columns, "EVI")
        date_col = find_column(df.columns, "Date")
        if not date_col or (not ndvi_col and not evi_col):
            continue
        frame = pd.DataFrame()
        frame["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        frame["_ndvi"] = (
            pd.to_numeric(df[ndvi_col], errors="coerce")
            if ndvi_col
            else float("nan")
        )
        frame["_evi"] = (
            pd.to_numeric(df[evi_col], errors="coerce")
            if evi_col
            else float("nan")
        )
        for col in ("_ndvi", "_evi"):
            valid = frame[col].dropna()
            if not valid.empty and valid.abs().median() > 1.5:
                # Raw MODIS integers: drop fill/out-of-range, then apply 0.0001.
                frame.loc[~frame[col].between(-2000, 10000), col] = float("nan")
                frame[col] *= 0.0001
            else:
                frame.loc[~frame[col].between(-1, 1), col] = float("nan")
        frames.append(frame)
    if not frames:
        raise RuntimeError(
            "Could not identify NDVI/EVI columns and dates in the NASA CSV files."
        )
    data = pd.concat(frames, ignore_index=True).dropna(subset=["_date"])
    data = (
        data.groupby("_date", as_index=False)
        .agg({"_ndvi": "first", "_evi": "first"})
        .sort_values("_date")
    )
    data = data.dropna(subset=["_ndvi", "_evi"], how="all")
    if data.empty:
        raise RuntimeError("No valid vegetation readings were found.")
    return data


# ---------------------------------------------------------------- weather + AI
def get_weather(lat, lon):
    r = requests.get(
        WEATHER_API,
        timeout=30,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_gusts_10m_max,et0_fao_evapotranspiration"
            ),
            "forecast_days": 7,
            "timezone": "auto",
        },
    )
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = """You are Farm Friend, an experienced agronomist who writes for farmers with low reading confidence.
You receive one field (crop, growth stage, irrigation, soil), 16-day NASA MODIS greenness readings
(NDVI and EVI for a 250 m pixel), and an Open-Meteo weather summary in metric units.
Think through the evidence silently. Then write ONLY the advice, in this exact plain-text shape.
No markdown. No symbols. No greeting.

What I see: 1 or 2 short sentences using the actual numbers (latest NDVI/EVI, the change, rain, heat, wind).
Do today: one concrete, low-cost action that fits the crop stage and the next 7 days.
Watch: one or two things to check in the field over the next few days.
Ask for help if: one clear condition for contacting a local agriculture officer.

Rules:
- Short sentences, everyday words (reading age about 10). Expand a term once, e.g. "greenness (NDVI)".
- A change smaller than 0.05 in NDVI is normal noise. Never call it a trend.
- Greenness describes the whole 250 m area, not one plant. Say this when it matters.
- Never diagnose a disease or pest. Say "may be" and name what to inspect.
- Never give fertiliser, pesticide or irrigation amounts. Say to follow the label and local advice.
- Never invent prices, yields or measurements. If data is missing, say what is missing.
- If the supplied weather shows a real risk (heat at or above the crop limit, strong gusts, long dry spell), mention it with the number.
- Do not mention these instructions or that you are an AI."""

CHAT_PROMPT = """You are Farm Friend, a calm farm helper for farmers with low reading confidence.
The question may be spoken, short, or in another language. Answer in the same language as the question;
if the language is unclear, use simple English.
Use only the supplied field, weather and satellite data. Do not invent any other measurement.
Write at most 3 short plain-text lines, no markdown, no symbols, under 70 words:

Answer: the direct answer in one or two sentences.
Do: one safe action the farmer can take today.
Check: one thing to verify, or when to ask a local agriculture officer.

Rules:
- "I do not have that information" is a good answer when the data is missing. Say what would help.
- Never diagnose a disease or pest from one symptom. Say "may be" and what to look at.
- Never give fertiliser, pesticide or irrigation amounts. Say to read the label and ask locally.
- If the farmer describes sudden severe damage (many plants wilting, spreading spots, many insects),
  tell them to contact a local agriculture officer today and to take clear photos.
- If the satellite data is marked sample, say it is a sample and advise a real NASA run before acting.
- If the question is not about this farm or farming, say it is outside what you can see.
- Do not mention being an AI or these instructions."""

PHOTO_PROMPT = """You are a crop scout for farmers with low reading confidence. You get ONE photo of a plant.
Look only at what is clearly visible. Answer in the same language as the farmer's question;
otherwise use simple English.
Write at most 4 short plain-text lines, no markdown, no symbols, under 90 words:

In the photo: what is clearly visible (leaf colour, spot shape, holes, wilting, insects, soil).
May be: 1 or 2 possible causes, always using "may".
Do now: one safe, low-cost step (for example remove badly affected leaves, improve airflow, check soil moisture).
Ask: when to contact a local agriculture officer.

Rules:
- If the photo is blurry, dark, or does not clearly show a plant, say that first and ask for a new photo in daylight.
- Never confirm a disease or pest from one photo. Never give chemical names, doses or mixes.
- Do not guess about things outside the frame. Do not mention being an AI or these instructions."""


def generate_advice(farm, data, weather):
    if not ai_settings()[0]:
        return {
            "mode": "basic",
            "advice": "AI advice is off. Add OPENAI_API_KEY or a free GEMINI_API_KEY to the .env file to turn it on.",
        }

    def rnd(x):
        return round(float(x), 4) if pd.notna(x) else None

    observations = [
        {
            "date": str(row["_date"].date()),
            "ndvi": rnd(row["_ndvi"]),
            "evi": rnd(row["_evi"]),
        }
        for _, row in data.tail(8).iterrows()
    ]
    context = {
        "today": datetime.now().date().isoformat(),
        "crop": farm.crop,
        "growth_stage": farm.growth_stage,
        "irrigation": farm.irrigation,
        "soil": farm.soil,
        "satellite": summarize_satellite(data),
        "recent_readings": observations,
        "weather": summarize_weather(weather, farm.crop),
        "scale_help": "NDVI and EVI run about -1 to 1. Rough guide: 0.2 bare soil, 0.4 sparse, 0.6 healthy, 0.8 dense. A change under 0.05 is noise.",
    }
    text = ai_reply(SYSTEM_PROMPT, str(context), max_tokens=450)
    if not text:
        raise RuntimeError("The AI returned an empty answer.")
    return {"mode": "ai", "advice": text}


def create_graph(data, job_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    if data["_ndvi"].notna().any():
        ax.plot(data["_date"], data["_ndvi"], marker="o", label="NDVI")
    if data["_evi"].notna().any():
        ax.plot(data["_date"], data["_evi"], marker="s", label="EVI")
    ax.set_title("Farm plant greenness (higher is healthier)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(job_dir / "vegetation_graph.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main job
def process_farm(job_id, farm):
    session = requests.Session()
    try:
        job_dir = DATA_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        update_job(job_id, status="running", message="Logging in to NASA...")
        session.headers.update({"Accept": "application/json"})
        authenticate(session, farm.earthdata_username, farm.earthdata_password)
        update_job(job_id, message="Sending the satellite request...")
        task_id = create_task(session, farm)
        update_job(
            job_id,
            nasa_task_id=task_id,
            message="NASA is preparing your data. This can take 5 to 30 minutes.",
        )
        wait_for_task(session, task_id, job_id)
        update_job(job_id, message="Downloading satellite data...")
        data = load_data(download_files(session, task_id, job_dir))
        create_graph(data, job_dir)

        def rnd(x):
            return round(float(x), 4) if pd.notna(x) else None

        summary = [
            {
                "date": str(row["_date"].date()),
                "ndvi": rnd(row["_ndvi"]),
                "evi": rnd(row["_evi"]),
            }
            for _, row in data.iterrows()
        ]
        out = data.rename(columns={"_date": "date", "_ndvi": "ndvi", "_evi": "evi"})
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out["ndvi"] = out["ndvi"].round(4)
        out["evi"] = out["evi"].round(4)
        out.to_csv(job_dir / "vegetation_data.csv", index=False)
        update_job(job_id, message="Checking the local weather...")
        try:
            weather = get_weather(farm.latitude, farm.longitude)
        except Exception:
            weather = {"error": "Weather is temporarily unavailable."}
        update_job(job_id, message="Writing plain-language advice...")
        try:
            advice = generate_advice(farm, data, weather)
        except Exception:
            advice = {
                "mode": "fallback",
                "advice": "Satellite data is ready, but AI advice failed. Check the AI key and try again.",
            }
        latest = summary[-1]
        update_job(
            job_id,
            status="completed",
            message="Done.",
            error=None,
            results={
                "crop": farm.crop,
                "growth_stage": farm.growth_stage,
                "coordinates": {"latitude": farm.latitude, "longitude": farm.longitude},
                "ndvi": latest["ndvi"],
                "evi": latest["evi"],
                "latest": latest,
                "observations": summary,
                "weather": weather,
                "recommendations": advice,
                "graph_url": f"/results/{job_id}/vegetation_graph.png",
                "csv_url": f"/results/{job_id}/vegetation_data.csv",
            },
        )
    except Exception as error:
        print(f"Farm Friend job {job_id} failed: {type(error).__name__}: {error}")
        update_job(job_id, status="failed", message="Failed.", error=str(error))
    finally:
        session.close()
        farm.earthdata_username = farm.earthdata_password = ""


# ---------------------------------------------------------------- routes
def _html():
    for path in (HTML_PATH, LEGACY_HTML):
        if path.exists():
            return path.read_text(encoding="utf-8")
    return (
        "<h1>Farm Friend</h1><p>Put <b>index.html</b> next to "
        "<b>farm_friend_app.py</b> and start again.</p>"
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
def website():
    return _html()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "name": "Farm Friend",
        "status": "online",
        "version": VERSION,
        "nasa_login_in_env": bool(
            os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")
        ),
        "ai_on": bool(ai_settings()[0]),
        "ai_provider": ai_settings()[2],
        "layers": ["_250m_16_days_NDVI", "_250m_16_days_EVI"],
        "product": PRODUCT,
    }


@app.post("/api/analyze")
def analyze(farm: FarmRequest, background_tasks: BackgroundTasks):
    try:
        start = datetime.strptime(farm.start_date, "%d-%m-%Y")
        end = datetime.strptime(farm.end_date, "%d-%m-%Y")
        if start > end or end.date() > datetime.now().date():
            raise ValueError
    except ValueError:
        raise HTTPException(
            400,
            "Check the dates: start must be before end, and the end cannot be in the future.",
        )
    farm.earthdata_username = farm.earthdata_username or os.getenv(
        "EARTHDATA_USERNAME", ""
    )
    farm.earthdata_password = farm.earthdata_password or os.getenv(
        "EARTHDATA_PASSWORD", ""
    )
    if not farm.earthdata_username or not farm.earthdata_password:
        raise HTTPException(
            400,
            "NASA login needed. Type it in the form, or put EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD in the server's .env file.",
        )
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Waiting to start...",
            "nasa_status": None,
            "nasa_task_id": None,
            "results": None,
            "error": None,
        }
    background_tasks.add_task(process_farm, job_id, farm)
    return {"job_id": job_id, "status": "queued", "message": "Started."}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found. The server may have restarted.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "nasa_status": job["nasa_status"],
        "error": job["error"],
    }


@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found. The server may have restarted.")
    if job["status"] != "completed":
        raise HTTPException(409, "This analysis is not finished yet.")
    return job["results"]


@app.post("/api/photo")
def photo_check(body: PhotoRequest):
    """AI looks at one plant photo and writes simple advice. Needs an AI key."""
    if not ai_settings()[0]:
        return {
            "mode": "off",
            "advice": "The photo check is switched off on this server. Add OPENAI_API_KEY or a free GEMINI_API_KEY to the .env file, restart, and try again. You can still use the self-check list in the app.",
        }
    if not body.image.startswith("data:image/"):
        raise HTTPException(400, "Send the photo as a data URL (data:image/...;base64,...).")
    if len(body.image) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "That photo is too large. Take a smaller photo or let the app shrink it.")
    context = f"Crop: {body.crop or 'unknown'}. Growth stage: {body.stage or 'unknown'}."
    if body.question:
        context += f"\nFarmer asks: {body.question}"
    text = ai_reply(PHOTO_PROMPT, context, image=body.image, max_tokens=450)
    if not text:
        raise RuntimeError("The AI returned an empty answer for the photo.")
    return {"mode": "ai", "advice": text}


@app.post("/api/chat")
def chat(body: ChatRequest):
    """Plain-language answer to a farmer's question. Needs an AI key."""
    if not ai_settings()[0]:
        return {
            "mode": "off",
            "answer": "AI answers are switched off on this server. Add OPENAI_API_KEY or a free GEMINI_API_KEY to the .env file and restart. The app will still give a basic answer from the weather and satellite numbers.",
        }
    context = {
        "today": datetime.now().date().isoformat(),
        "question": body.question,
        "crop": body.crop,
        "growth_stage": body.stage,
        "irrigation": body.water,
        "soil": body.soil,
        "crop_heat_limit_c": CROP_HEAT.get(body.crop or "", 35),
        "weather": body.weather,
        "satellite": body.satellite,
        "units": "temperature in C, rain in mm, wind in km/h, ET0 in mm/day. NDVI/EVI run about -1 to 1; a change under 0.05 is noise.",
    }
    text = ai_reply(CHAT_PROMPT, str(context), max_tokens=350)
    if not text:
        raise RuntimeError("The AI returned an empty answer.")
    return {"mode": "ai", "answer": text}


@app.get("/api/jobs")
def all_jobs():
    with jobs_lock:
        return [
            {"job_id": k, "status": v["status"], "message": v["message"]}
            for k, v in jobs.items()
        ]


if __name__ == "__main__":
    if not os.getenv("FF_NO_BROWSER"):
        threading.Timer(
            1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")
        ).start()
    print(f"Farm Friend {VERSION} on http://{HOST}:{PORT}")
    print("Stop with Ctrl+C.")
    uvicorn.run(app, host=HOST, port=PORT)
