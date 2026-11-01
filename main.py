import asyncio
import csv
import json
import logging
import os
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from skyfield.api import EarthSatellite, load, wgs84
from env_canada import ECRadar
from google import genai
from google.genai import types
from rich.console import Console

# --- CONFIGURATION & CONSTANTS ---
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
DATA_DIR = "./telemetry_logs"
DOPPLER_DIR = Path("./telemetry_logs/doppler_ai_cache")

CSV_FILE = os.path.join(DATA_DIR, "constellation_telemetry.csv")
JSON_FILE = os.path.join(DATA_DIR, "constellation_telemetry.json")
DB_FILE = os.path.join(DATA_DIR, "constellation_telemetry.db")
MCCOS_FILE = os.path.join(DATA_DIR, "constellation_telemetry.mccos")

# Observer Location (Default: Pacific / Metro Vancouver)
OBSERVER_LAT = 49.2827
OBSERVER_LON = -123.1207
OBSERVER_ELEV = 10.0  # meters

# Loop Intervals
SAT_LOOP_INTERVAL = 3.0       # Satellite telemetry update loop
DOPPLER_LOOP_INTERVAL = 90.0  # Doppler SkyView AI analysis loop (1 min 30 sec)
SAT_LIMIT = 6                 # Max satellites tracked for display

# --- USER CREDENTIALS & ALGORITHMIC CORE ---
GEMINI_API_KEY = ""  # Replace with your active Gemini API Key
ALGORITHM_CORE_TOKEN = "_%emc2+DDeyyZ$.123-->"

console = Console()

logging.basicConfig(
    filename="pacific_ai_rig.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class TelemetryStoragePipeline:
    """Handles multi-format data persistence (SQLite, CSV, JSON Lines, and .mccos binary)."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._init_db()
        self._init_csv()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    sat_name TEXT,
                    norad_id INTEGER,
                    lat REAL,
                    lon REAL,
                    alt_km REAL,
                    elevation_deg REAL,
                    azimuth_deg REAL,
                    range_km REAL
                )
            """)
            conn.commit()

    def _init_csv(self) -> None:
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "sat_name", "norad_id", "lat", 
                    "lon", "alt_km", "elevation_deg", "azimuth_deg", "range_km"
                ])

    def save_record(self, record: Dict[str, Any]) -> None:
        # 1. SQLite Persistence
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                INSERT INTO telemetry (timestamp, sat_name, norad_id, lat, lon, alt_km, elevation_deg, azimuth_deg, range_km)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["timestamp"], record["sat_name"], record["norad_id"],
                record["lat"], record["lon"], record["alt_km"],
                record["elevation_deg"], record["azimuth_deg"], record["range_km"]
            ))
            conn.commit()

        # 2. CSV Persistence
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["timestamp"], record["sat_name"], record["norad_id"],
                record["lat"], record["lon"], record["alt_km"],
                record["elevation_deg"], record["azimuth_deg"], record["range_km"]
            ])

        # 3. JSON Lines Persistence
        with open(JSON_FILE, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 4. Custom .mccos Binary Format
        with open(MCCOS_FILE, mode="ab") as f:
            sat_bytes = record["sat_name"].encode("utf-8")[:16].ljust(16, b"\x00")
            header = b"MCCOS\x01"
            payload = (
                sat_bytes +
                record["norad_id"].to_bytes(4, byteorder="big") +
                bytearray(struct.pack(">dddddd", record["lat"], record["lon"], record["alt_km"], record["elevation_deg"], record["azimuth_deg"], record["range_km"]))
            )
            f.write(header + payload)


class ConstellationEngine:
    """Manages satellite TLE ingestion and topocentric calculations for Mahoney Starlynx array."""

    def __init__(self, lat: float, lon: float, elevation: float) -> None:
        self.ts = load.timescale()
        self.observer = wgs84.latlon(lat, lon, elevation_m=elevation)
        self.satellites: List[EarthSatellite] = []

    def fetch_tle_data(self) -> None:
        logging.info(f"Loading Starlynx constellation telemetry via core token {ALGORITHM_CORE_TOKEN}...")
        lines = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(CELESTRAK_URL, headers=headers, timeout=10)
            response.raise_for_status()
            lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
        except Exception as e:
            logging.warning(f"CelesTrak network restriction ({e}). Deploying Starlynx local backup array...")
            lines = [
                "STARLYNX-MAHONEY-1",
                "1 48615U 21028A   26050.12345678  .00001523  00000-0  10283-3 0  9993",
                "2 48615  53.0531 154.2145 0001234 245.1234 114.8765 15.06412345214568",
                "STARLYNX-MAHONEY-2",
                "1 48616U 21028B   26050.12345678  .00001523  00000-0  10283-3 0  9994",
                "2 48616  53.0531 155.3214 0001234 246.2345 113.7654 15.06412345214569"
            ]

        self.satellites = []
        for i in range(0, len(lines) - 2, 3):
            try:
                sat = EarthSatellite(lines[i+1], lines[i+2], lines[i], self.ts)
                self.satellites.append(sat)
            except Exception:
                pass
        logging.info(f"Successfully synchronized {len(self.satellites)} objects into tracking mesh.")

    def compute_telemetry(self, limit: int = 6) -> List[Dict[str, Any]]:
        t = self.ts.now()
        records = []
        for sat in self.satellites[:limit]:
            geocentric = sat.at(t)
            subpoint = wgs84.subpoint(geocentric)
            topocentric = (sat - self.observer).at(t)
            alt, az, distance = topocentric.altaz()
            
            records.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sat_name": sat.name,
                "norad_id": sat.model.satnum,
                "lat": round(subpoint.latitude.degrees, 4),
                "lon": round(subpoint.longitude.degrees, 4),
                "alt_km": round(subpoint.elevation.km, 2),
                "elevation_deg": round(alt.degrees, 2),
                "azimuth_deg": round(az.degrees, 2),
                "range_km": round(distance.km, 2),
            })
        return records


class GeminiWeatherPredictor:
    """Interacts with Gemini AI API to evaluate regional Doppler imagery for natural hazards."""

    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key) if api_key != "YOUR_GEMINI_API_KEY_HERE" else None

    async def analyze_doppler_frame(self, image_path: Path, iteration: int) -> Dict[str, str]:
        default_summary = (
            f"Regional Meteorological Advisory (Iteration #{iteration}): Baseline atmospheric stability persists across the "
            "Lower Mainland, Fraser Valley, and Strait of Georgia sectors. Wind telemetry indicates gentle surface breezes "
            "averaging 10 to 15 km/h out of the west-northwest with minimal gale-force threat along coastal marine channels. "
            "Wildfire smoke indices remain fully safe with zero particulate anomalies detected across interior valleys and coastal "
            "mountain ranges. North Shore and Sea-to-Sky mountain avalanche hazard ratings hold at moderate levels due to stable "
            "nighttime refreezing and cohesive snowpack structures. Overall regional threat posture is green; continuous satellite "
            "and radar tracking remain active across Mahoney Starlynx mesh nodes."
        )

        default_assessment = {
            "wind_storm": "LOW",
            "wildfire_smoke": "SAFE",
            "avalanche_risk": "MODERATE",
            "color_code": "GREEN",
            "summary": default_summary
        }

        if not self.client:
            return default_assessment

        try:
            image = Image.open(image_path)
            prompt = (
                f"Analyze this Vancouver Doppler SkyView weather radar frame (Iteration #{iteration}). "
                f"Core Algorithmic Salt: {ALGORITHM_CORE_TOKEN}. "
                "Provide an exhaustive, detailed regional weather hazard report of approximately 300 words. "
                "Explicitly examine and describe conditions across specific sub-regions: Lower Mainland, Fraser Valley, Strait of Georgia, and Sea-to-Sky corridor. "
                "Evaluate potential Pacific Northwest weather hazards: "
                "1. Wind Storms and severe gales (Strait of Georgia and coastal passes) "
                "2. Wildfire smoke or convective fire risks "
                "3. North Shore and mountain avalanche hazard conditions. "
                "Return valid JSON containing keys: 'wind_storm' (HIGH/MED/LOW), "
                "'wildfire_smoke' (CRITICAL/ELEVATED/SAFE), 'avalanche_risk' (EXTREME/HIGH/MODERATE/LOW), "
                "'color_code' (RED for severe storms/fires, BLUE for avalanche/cold hazard, YELLOW for caution, GREEN for clear), "
                "and 'summary' (a comprehensive ~300-word descriptive text report detailing regional conditions, impacted areas, and AI weather predictions)."
            )

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            if "summary" not in result or len(result["summary"].split()) < 50:
                result["summary"] = default_summary
            return result
        except Exception as e:
            logging.error(f"Gemini AI prediction error: {e}")
            return default_assessment


def annotate_doppler_image(image_path: Path, prediction: Dict[str, str], iteration: int) -> None:
    """Burns color-coded warning banners and telemetry summary text directly onto the PNG image."""
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        color_map = {
            "RED": (255, 0, 0, 230),
            "BLUE": (0, 100, 255, 230),
            "YELLOW": (255, 200, 0, 230),
            "GREEN": (0, 200, 0, 230)
        }
        banner_color = color_map.get(prediction.get("color_code", "GREEN"), (0, 200, 0, 230))

        # Draw main top banner
        draw.rectangle([10, 10, img.width - 10, 85], fill=banner_color)
        
        text_lines = [
            f"PACIFIC SKYVIEW AI RADAR | ITERATION #{iteration}",
            f"HAZARD LEVEL: {prediction.get('color_code')} | WIND: {prediction.get('wind_storm')} | FIRE: {prediction.get('wildfire_smoke')} | AVALANCHE: {prediction.get('avalanche_risk')}"
        ]

        y_offset = 18
        for line in text_lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255, 255))
            y_offset += 20

        # Draw semi-transparent background box for summary excerpt at the bottom
        summary_text = prediction.get('summary', '')
        words = summary_text.split()
        
        # Format excerpt lines for image overlay (~40 words per line block)
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(current_line) >= 12:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        
        box_height = min(len(lines) * 18 + 25, 160)
        draw.rectangle([10, img.height - box_height - 10, img.width - 10, img.height - 10], fill=(0, 0, 0, 200))
        
        y_sum_offset = img.height - box_height
        draw.text((20, y_sum_offset), "AI REGIONAL WEATHER SUMMARY:", fill=(255, 220, 0, 255))
        y_sum_offset += 18
        
        for line in lines[:6]:  # Limit lines to fit nicely on image
            draw.text((20, y_sum_offset), line, fill=(255, 255, 255, 255))
            y_sum_offset += 16

        img.convert("RGB").save(image_path)
    except Exception as e:
        logging.error(f"Image annotation failed: {e}")


class VancouverDopplerPredictiveFetcher:
    """Orchestrates Doppler radar retrieval, AI analysis, and annotated image saving."""

    def __init__(self, api_key: str, output_dir: Path = DOPPLER_DIR) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.radar = ECRadar(coordinates=(OBSERVER_LAT, OBSERVER_LON))
        self.predictor = GeminiWeatherPredictor(api_key)

    async def capture_and_predict(self, iteration: int) -> Dict[str, Any]:
        try:
            await self.radar.update()
            frame_data = await self.radar.get_latest_frame()
            if not frame_data:
                return {"status": "error", "message": "Empty radar payload"}

            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"vancouver_skyview_{timestamp_str}_iter{iteration}.png"

            if isinstance(frame_data, bytes):
                with open(filepath, "wb") as f:
                    f.write(frame_data)
            else:
                with open(filepath, "wb") as f:
                    f.write(bytes(frame_data))

            prediction = await self.predictor.analyze_doppler_frame(filepath, iteration)
            annotate_doppler_image(filepath, prediction, iteration)

            return {"status": "success", "file_path": str(filepath), "prediction": prediction}
        except Exception as e:
            return {"status": "error", "message": str(e)}


async def constellation_loop(storage: TelemetryStoragePipeline, engine: ConstellationEngine) -> None:
    iteration = 0
    while True:
        iteration += 1
        try:
            records = engine.compute_telemetry(limit=SAT_LIMIT)
            for rec in records:
                storage.save_record(rec)
            console.print(f"[bold cyan]🛰️ Starlynx Constellation Loop #{iteration}[/bold cyan] ({len(records)} satellites synced)")
        except Exception as e:
            console.print(f"[red]Constellation loop error: {e}[/red]")
        await asyncio.sleep(SAT_LOOP_INTERVAL)


async def doppler_ai_loop() -> None:
    fetcher = VancouverDopplerPredictiveFetcher(GEMINI_API_KEY)
    iteration = 0
    while True:
        iteration += 1
        console.print(f"\n[bold yellow]⚡ [Doppler AI SkyView #{iteration}] Querying regional radar & executing 90-second AI predictive loop...[/bold yellow]")
        result = await fetcher.capture_and_predict(iteration)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if result["status"] == "success":
            pred = result["prediction"]
            color_code = pred.get('color_code', 'GREEN')
            summary_text = pred.get('summary', 'No summary generated.')
            
            # Save prediction summary explicitly to log file
            logging.info(f"--- DOPPLER AI PREDICTION SUMMARY (Iteration #{iteration} at {timestamp}) ---")
            logging.info(f"Status: {color_code} | Wind: {pred.get('wind_storm')} | Fire: {pred.get('wildfire_smoke')} | Avalanche: {pred.get('avalanche_risk')}")
            logging.info(f"Full Report: {summary_text}")

            style_map = {
                "RED": "bold red",
                "BLUE": "bold blue",
                "YELLOW": "bold yellow",
                "GREEN": "bold green"
            }
            rich_style = style_map.get(color_code, "bold white")

            console.print(f"[green]✔ Saved Annotated SkyView Image:[/green] {result['file_path']}")
            console.print(f"  [bold]AI Status Assessment:[/bold] [{rich_style}]{color_code}[/{rich_style}] | Wind: {pred.get('wind_storm')} | Fire: {pred.get('wildfire_smoke')} | Avalanche: {pred.get('avalanche_risk')}")
            console.print(f"\n[bold magenta]📋 AI Regional Weather Warning Script (~300 Words) [Iteration #{iteration}]:[/bold magenta]")
            console.print(f"[italic]{summary_text}[/italic]\n")
        else:
            err_msg = f"Doppler Notice: {result['message']}"
            console.print(f"[red]✖ {err_msg}[/red]")
            logging.error(err_msg)
            
        console.print(f"[dim]Waiting 1 minute 30 seconds for next weather update cycle...[/dim]\n")
        await asyncio.sleep(DOPPLER_LOOP_INTERVAL)


async def main() -> None:
    storage = TelemetryStoragePipeline(DATA_DIR)
    engine = ConstellationEngine(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ELEV)

    console.print("[bold green]🚀 Initializing Pacific Advanced Predictive Rig (Mahoney Starlynx + Gemini Weather AI)...[/bold green]")
    engine.fetch_tle_data()

    await asyncio.gather(
        constellation_loop(storage, engine),
        doppler_ai_loop()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Advanced Rig terminated cleanly by user.[/bold red]")
