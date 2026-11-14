"""Farm Friend backend (Python).

One file, next to index.html. Serves the app and adds:
  - NASA AppEEARS satellite analysis  (MOD13Q1.061, NDVI + EVI, 250 m, 16 days)
  - Open-Meteo weather
  - OpenAI plain-language advice, chat and plant-photo check

Run:      python farm_friend_app.py
Open:     http://127.0.0.1:8000      (the app opens by itself)

Optional .env file next to this script:
  OPENAI_API_KEY=sk-...
  EARTHDATA_USERNAME=your_nasa_login
  EARTHDATA_PASSWORD=your_nasa_password
  OPENAI_MODEL=gpt-4.1-mini          (optional)
  PORT=8000                          (optional)

Install:  pip install -r requirements.txt
"""
import os
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
VERSION = "5.0.0"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MAX_IMAGE_CHARS = 12_000_000  # about 9 MB of image data

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


SYSTEM_PROMPT = """You are a friendly farm adviser for farmers who may not read well.
Use the satellite greenness readings (NDVI, EVI) and the weather to help.
Write VERY short sentences and simple everyday words. No jargon.
Use at most 8 bullet points, each starting with what to DO.
Cover: how the plants look now, whether they are getting better or worse, watering,
soil and food for the plants, what to check in the field, and what extra information would help.
Rules: say what you SEE (readings) separately from what you GUESS. Never name a disease
from NDVI/EVI alone. Never invent yield numbers. Never give fertiliser or pesticide amounts;
say to read the product label and ask a local agriculture officer. Say when you are unsure.
Tell the farmer to check the field."""

CHAT_PROMPT = """You are Farm Friend, a calm farm helper for farmers who may not read well.
Answer the farmer's question in at most 6 short bullet points. Start each bullet with a verb.
Use simple everyday words. No jargon. If numbers are supplied, use them in plain words.
Say what you SEE separately from what you GUESS. Never diagnose a disease from one symptom.
Never give fertiliser or pesticide amounts. Tell the farmer to read the product label and ask
a local agriculture officer. If the question needs information you do not have, say so and say
what to check. Never invent prices, yields or measurements. Keep it under 130 words."""

PHOTO_PROMPT = """You are a crop helper for farmers who may not read well. You get ONE photo of a plant.
Write VERY short sentences, simple everyday words, no jargon. At most 6 bullet points, each
starting with what to DO (for example: "Water less.", "Look under the leaves."). Describe only
what is clearly visible in the photo (leaf colour, spots, holes, wilting, insects, soil). If the
photo is blurry or the plant is not clear, say so and tell the farmer to take a new photo in daylight.
Never be sure about a disease from one photo; say "may be" and tell the farmer to ask a local
agriculture officer. Never give fertiliser or pesticide amounts. Answer the farmer's question if
there is one."""


def generate_advice(farm, data, weather):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "mode": "basic",
            "advice": "AI advice is off. Add OPENAI_API_KEY to the .env file to turn it on.",
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
        "crop": farm.crop,
        "growth_stage": farm.growth_stage,
        "irrigation": farm.irrigation,
        "soil": farm.soil,
        "observations": observations,
        "weather": weather,
    }
    r = requests.post(
        OPENAI_API,
        timeout=90,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(context)},
            ],
        },
    )
    r.raise_for_status()
    try:
        return {"mode": "ai", "advice": r.json()["choices"][0]["message"]["content"]}
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("OpenAI returned an unexpected answer.")


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
                "advice": "Satellite data is ready, but AI advice failed. Check OPENAI_API_KEY and try again.",
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
        "ai_on": bool(os.getenv("OPENAI_API_KEY")),
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
    """AI looks at one plant photo and writes simple advice. Needs OPENAI_API_KEY."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "mode": "off",
            "advice": "The photo check is switched off on this server. Add OPENAI_API_KEY to the .env file, restart, and try again. You can still use the self-check list in the app.",
        }
    if not body.image.startswith("data:image/"):
        raise HTTPException(400, "Send the photo as a data URL (data:image/...;base64,...).")
    if len(body.image) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "That photo is too large. Take a smaller photo or let the app shrink it.")
    context = f"Crop: {body.crop or 'unknown'}. Growth stage: {body.stage or 'unknown'}."
    if body.question:
        context += f"\nFarmer asks: {body.question}"
    r = requests.post(
        OPENAI_API,
        timeout=90,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": PHOTO_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {"type": "image_url", "image_url": {"url": body.image}},
                    ],
                },
            ],
        },
    )
    r.raise_for_status()
    try:
        return {"mode": "ai", "advice": r.json()["choices"][0]["message"]["content"]}
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("OpenAI returned an unexpected answer for the photo.")


@app.post("/api/chat")
def chat(body: ChatRequest):
    """Plain-language answer to a farmer's question. Needs OPENAI_API_KEY."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "mode": "off",
            "answer": "AI answers are switched off on this server. Add OPENAI_API_KEY to the .env file and restart. The app will still give a basic answer from the weather and satellite numbers.",
        }
    context = {
        "question": body.question,
        "crop": body.crop,
        "growth_stage": body.stage,
        "irrigation": body.water,
        "soil": body.soil,
        "weather": body.weather,
        "satellite": body.satellite,
    }
    r = requests.post(
        OPENAI_API,
        timeout=90,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": str(context)},
            ],
        },
    )
    r.raise_for_status()
    try:
        return {"mode": "ai", "answer": r.json()["choices"][0]["message"]["content"]}
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("OpenAI returned an unexpected answer.")


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
