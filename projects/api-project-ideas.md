# API Project Ideas

Seven project concepts combining free/public APIs across space, aviation, weather, government, ML, and maps. Auth levels: **[FREE]** = no auth, **[KEY]** = free API key, **[OAuth]** = OAuth flow.

---

## 1. Real-Time Flight & Weather Command Center

**What it does:** A live web dashboard showing every aircraft in a chosen region on a map, overlaid with aviation weather (METARs, TAFs, SIGMETs). Click a plane → see altitude, speed, heading, origin/destination. Click an airport → see current conditions and forecast.

**APIs:**
- **ADS-B Exchange** — live aircraft positions (lat/lng, altitude, speed, callsign, squawk)
- **AviationWeather (NOAA)** — METARs, TAFs, SIGMETs, PIREPs
- **Open-Meteo** — general forecast for non-airport areas
- **OpenStreetMap / Mapbox** — base map tiles
- **airportsapi** — resolve ICAO codes to airport names

**Core features:**
- Live-updating map (poll every 5–10s)
- Filter by altitude, airline, aircraft type
- Click plane → flight info card
- Color-coded weather overlay (green/yellow/red by METAR conditions)
- SIGMET polygons drawn on map (turbulence, icing zones)
- Search by callsign or registration

**Tech stack:** React + Leaflet (or Mapbox GL) + small Node/Python backend for caching. WebSocket for live updates.

**Difficulty:** Medium-hard. Hard part is performance — rendering 500+ planes smoothly without hammering APIs.

**Why it's cool:** Basically a free Flightradar24 with pilot-grade weather. Genuinely useful, not just a demo.

---

## 2. ISS Overhead Alert System

**What it does:** Tells you exactly when the ISS will pass over your location AND the sky will be clear enough to see it. Sends a notification when conditions align.

**APIs:**
- **Open Notify** — current ISS position + pass predictions
- **Open-Meteo** — cloud cover forecast
- **7Timer!** — astro-specific weather (seeing, transparency, darkness)
- **NASA APOD** — bonus astronomy picture in the notification

**Core features:**
- Enter your location once (or browser geolocation)
- Backend cron job checks daily for upcoming passes
- For each pass, check cloud cover at that exact time
- If cloud cover < 30% and pass is visible (not daytime), trigger alert
- Web dashboard showing next 7 days of passes with viewing quality scores
- Optional: email/Telegram/Discord webhook notifications

**Tech stack:** Python + FastAPI + simple frontend. APScheduler for periodic checks. SQLite for prefs and pass history.

**Difficulty:** Easy-medium. Weekend project.

**Why it's cool:** Solves a real problem. Genuinely shippable as a tiny SaaS.

---

## 3. Government Transparency Dashboard

**What it does:** Visualization tool that shows federal regulations being published, cross-referenced with the demographics of who they affect.

**APIs:**
- **Federal Register** — daily federal rules, proposed rules, agency notices
- **Census.gov** — demographic data by county/state
- **Data USA** — economic and education data
- **US Presidential Election Data** — voting patterns by region (optional)

**Core features:**
- Daily feed of new federal rules, categorized by agency and topic
- Click a rule → NLP extracts affected industries/regions
- Overlay on US map with demographic shading
- Filter by agency (EPA, FDA, DOL, etc.)
- "Impact score" combining population affected + economic exposure
- Historical trend: regulations per agency over time

**Tech stack:** Python (pandas) + FastAPI + React + D3.js / Observable Plot. PostgreSQL for the regulation corpus.

**Difficulty:** Hard. Data modeling is the tough part — mapping regulation text to "who it affects" requires NLP or manual tagging.

**Why it's cool:** Civic-tech, portfolio-worthy, the kind of thing journalists and policy nerds want.

---

## 4. Space Launch Tracker on a 3D Globe

**What it does:** A spinning 3D Earth showing every upcoming rocket launch as a glowing pin. Click a launch → countdown timer, mission details, weather at the launch site, go/no-go forecast.

**APIs:**
- **Launch Library 2** — every upcoming launch worldwide
- **Open-Meteo** — weather forecast at launch coordinates
- **AviationWeather** — winds aloft at launch site
- **NASA** — mission imagery and details
- **Mapbox** or **CesiumJS** — 3D globe rendering

**Core features:**
- Interactive 3D globe (CesiumJS recommended)
- Glowing pins at active launch sites
- Live countdown timers per launch
- Click pin → mission card with rocket photo, payload, weather forecast
- "Launch weather score" — green/yellow/red
- Past launches mode (animate through history)
- Filter by agency (SpaceX, NASA, ISRO, ESA, Roscosmos, etc.)

**Tech stack:** React + CesiumJS + small Node backend for caching. No database needed.

**Difficulty:** Medium. CesiumJS has a learning curve but is well-documented.

**Why it's cool:** Visually stunning. Portfolio gold. Space launches happen 2–3x/week, so always fresh.

---

## 5. Carbon-Aware Flight Planner

**What it does:** Show real-time aircraft positions, but each plane is colored by its carbon footprint per passenger. Pick two airports → estimate CO2 of that route → suggest offset cost.

**APIs:**
- **ADS-B Exchange** — live aircraft + aircraft type (key for fuel burn estimation)
- **UK Carbon Intensity** — grid carbon for comparison
- **Open-Meteo** — winds aloft (affects fuel burn)
- **CO2 Offset** — calculate offset cost
- **airportsapi** — airport name resolution

**Core features:**
- Live map of flights, colored by estimated CO2/passenger-km
- Route planner: pick origin + destination → estimate emissions
- Comparison mode: "this flight = X car trips = Y months of household electricity"
- Offset calculator with real provider links
- Aggregate stats: "today, X tons of CO2 emitted by aviation in [region]"

**Tech stack:** React + Leaflet + Python backend (pandas for ICAO fuel burn lookup tables).

**Difficulty:** Medium-hard. Fuel burn modeling is the hard part.

**Why it's cool:** Climate-relevant, novel angle, less common than #1.

---

## 6. Asteroid Threat Map

**What it does:** Visualize all known near-Earth objects, their orbits, close approaches, and what would happen if one hit. Pick an asteroid → see size, speed, composition, theoretical impact zone, energy release.

**APIs:**
- **Minor Planet Center / Asterank** — asteroid orbital elements, sizes, compositions, value
- **NASA NEO API** — close-approach data
- **Newton** — orbital mechanics calculations
- **OpenStreetMap / Mapbox** — impact zone visualization

**Core features:**
- List of upcoming close approaches sorted by proximity/risk
- 3D solar system view showing asteroid orbits (Three.js)
- Click asteroid → details panel with size, mass, velocity
- "Impact simulator": pick a hit location → calculate crater size, blast radius, casualties (Earth Impact Effects formulas)
- Filter by hazardous (PHA) vs. non-hazardous
- Asterank's "value" mode: which asteroids are worth mining?

**Tech stack:** React + Three.js (orbits) + Leaflet (impact maps) + Python backend for orbital calcs.

**Difficulty:** Hard. Orbital mechanics is non-trivial.

**Why it's cool:** Astronomy + physics + geography + viz in one. Great conversation piece.

---

## 7. AI-Powered Satellite Image Analyzer

**What it does:** Pick any spot on Earth → fetch a recent satellite image → run computer vision on it → identify buildings, roads, vegetation, water, vehicles. Track changes over time.

**APIs:**
- **NASA Earth imagery** — Landsat satellite images by lat/lng/date
- **OpenVisionAPI** — open-source object detection (no key)
- **Open-Meteo** — weather context (was it cloudy that day?)
- **Mapbox** — base map for picking locations
- **OpenStreetMap (Nominatim)** — geocoding place names

**Core features:**
- Type address or click map → get satellite image
- Run object detection → annotated image with bounding boxes
- "Time machine" mode: same spot 1, 5, 10 years ago
- Diff visualization: highlight what changed
- Pre-built use cases: deforestation, coastal erosion, city growth
- Export reports as PDF

**Tech stack:** Python (Pillow/OpenCV) + FastAPI + React frontend. Sentinel Hub for higher resolution later.

**Difficulty:** Medium. NASA's Earth API is finicky (limited dates, cloudy images, low res). CV part is straightforward.

**Why it's cool:** Real-world AI application, environmentally meaningful, "time machine" demo is great.

---

## Quick Comparison

| # | Project | Difficulty | Wow Factor | Time to MVP | Best For |
|---|---------|-----------|------------|-------------|----------|
| 1 | Flight & Weather Command Center | Medium-hard | High | 1-2 weeks | Aviation nerds, dashboards |
| 2 | ISS Overhead Alert | Easy-medium | Medium | Weekend | Quick win, real users |
| 3 | Government Transparency | Hard | Medium-high | 2-3 weeks | Civic tech, data viz |
| 4 | Space Launch Tracker (3D) | Medium | Very high | 1-2 weeks | Visual portfolio piece |
| 5 | Carbon-Aware Flight Planner | Medium-hard | High | 2 weeks | Climate + novelty |
| 6 | Asteroid Threat Map | Hard | High | 2-3 weeks | Physics + viz lovers |
| 7 | Satellite Image Analyzer | Medium | High | 1-2 weeks | AI/CV portfolio |

---

## Recommendations

- **Quickest impressive win:** #2 (ISS Alert) or #4 (3D Launch Tracker)
- **Best portfolio piece:** #4 (3D Launch Tracker) — looks incredible
- **Most useful day-to-day:** #1 (Flight & Weather) or #2 (ISS)
- **Best for ML practice:** #7 (Satellite Analyzer)
- **Most ambitious:** #3 or #6
