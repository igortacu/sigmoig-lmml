# Smart Battlesnake (FastAPI)

An advanced, production‑ready Battlesnake agent built with **FastAPI**. It focuses on **space control (flood‑fill), safe pathing, food strategy, hazard avoidance,** and **opportunistic head‑to‑head wins**. Tuned to help you defeat the public **“sigmoid‑serpent”** three times in a row.

---

## Features
- 🧭 **Safe navigation:** Avoids walls, bodies, and hazards with strong static safety filters.
- 🧮 **Space heuristic:** Flood‑fill (reachable area) scoring to minimize self‑traps.
- 🍏 **Smart food logic:** Health‑aware + catch‑up growth targeting.
- 🐍 **Head‑to‑head awareness:** Takes winning trades when longer; avoids ties/losses.
- ⚙️ **Configurable aggression:** `AGGRESSION` env var to tune risk vs. safety.
- 🗣️ **Battle logging:** Turn‑by‑turn shout + console logs for quick debugging.

---

## Repository Layout
This project is a single‑file server for easy deployment:

```
./battlesnake_smart_serpent.py   # FastAPI server + game logic
```

> If you want a multi‑file layout (Dockerfile, Procfile, requirements.txt), see the **Deployment** section.

---

## Requirements
- Python 3.9+
- `fastapi`, `uvicorn`

Install:
```bash
pip install fastapi uvicorn
```

---

## Quickstart (Local)
Run the server on port 8000:
```bash
python battlesnake_smart_serpent.py
```
Expose it publicly for Battlesnake to reach your snake (choose one):
- **ngrok:** `ngrok http 8000`
- **Cloud deploy:** Render / Railway / Fly.io (see below)

Configure your Battlesnake dashboard to point to your public base URL.

### Health Check
Open your base URL in a browser. You should see JSON like:
```json
{
  "apiversion": "1",
  "author": "smart-serpent",
  "color": "#22c55e",
  "head": "beluga",
  "tail": "bolt"
}
```

---

## Environment Variables
| Name         | Type   | Default | Description |
|--------------|--------|---------|-------------|
| `PORT`       | int    | `8000`  | Web server port. |
| `LOG_LEVEL`  | str    | `INFO`  | `DEBUG`, `INFO`, `WARN`, `ERROR`. |
| `AGGRESSION` | float  | `1.0`   | >1 makes riskier head‑to‑heads; <1 safer. |

Usage example:
```bash
AGGRESSION=1.4 LOG_LEVEL=DEBUG python battlesnake_smart_serpent.py
```

---

## Battlesnake API Endpoints
This server implements the standard API:
- `GET /` – Info (appearance & version)
- `POST /start` – Game start
- `POST /move` – **Decision** (returns `{"move": "up|down|left|right"}`)
- `POST /end` – Game end

### Local Testing (sample `curl`)
```bash
# Info
curl localhost:8000/

# Move (example payload snippet)
curl -X POST localhost:8000/move \
  -H 'Content-Type: application/json' \
  -d '{
        "game": {"id": "test", "timeout": 500},
        "turn": 10,
        "board": {
          "height": 11, "width": 11,
          "food": [{"x": 5, "y": 5}],
          "hazards": [],
          "snakes": [
            {"id": "me","name": "me","health": 80,
             "body": [{"x": 3,"y": 3},{"x": 3,"y": 2},{"x": 3,"y": 1}],
             "head": {"x": 3,"y": 3}, "length": 3},
            {"id": "opp","name": "opp","health": 90,
             "body": [{"x": 7,"y": 7},{"x": 7,"y": 6},{"x": 7,"y": 5}],
             "head": {"x": 7,"y": 7}, "length": 3}
          ]
        },
        "you": {"id": "me","name": "me","health": 80,
                 "body": [{"x": 3,"y": 3},{"x": 3,"y": 2},{"x": 3,"y": 1}],
                 "head": {"x": 3,"y": 3}, "length": 3}
      }'
```

---

## Strategy & Heuristics (What Makes It “Smart”)
1. **Hard Safety Filters**  
   Reject moves that: go out of bounds, collide with any body, or enter cells that won’t be vacated.
2. **Tail Logic**  
   Treat our tail cell as free **if we won’t eat this turn** (tail moves forward).
3. **Space (Flood‑Fill Area)**  
   For each candidate move, compute reachable cell count with BFS. Bigger is better.
4. **Head‑to‑Head Scoring**  
   Nearby opponent heads penalize/boost the score depending on relative lengths. Favor trades when strictly longer.
5. **Food Targeting**  
   Dynamically seek food when low health or behind in size; otherwise prioritize space and position.
6. **Centering Prior**  
   A tiny bias away from corners/walls to reduce accidental box‑ins.
7. **Stability Bonus**  
   Slightly prefer continuing previous direction to reduce jitter.

> All pieces combine into a **single scalar score** per move. The highest‑scoring legal move is chosen each turn.

---

## Beating “sigmoid‑serpent”: Practical Tips
- **Increase controlled aggression:** `AGGRESSION=1.3–1.6` helps win contested cells.
- **Outlast starvation plays:** If the opponent starves you, raise the food seeking trigger (in code, look for `need_food` condition; try `my_health <= 50`).
- **More room to breathe:** Boost space weight (in `choose_move`, raise `score += 0.2 * area` to `0.30–0.35`).
- **Post‑game review:** Use the `shout` logs (e.g., `t37:right`) alongside the replay to correlate turns with choices.

Achieve **3 consecutive wins**, grab **screenshots or replay links**, and contact the admins per task rules.

---

## Deployment
Below are lightweight options. Pick one and point your Battlesnake to the resulting URL.

### Render (free tier)
1. Create a **New Web Service** → **Deploy from Git**.
2. Runtime: Python; Start command:
   ```bash
   uvicorn battlesnake_smart_serpent:app --host 0.0.0.0 --port $PORT
   ```
3. Add env vars: `AGGRESSION`, `LOG_LEVEL` as you like.

**requirements.txt** (example):
```txt
fastapi
uvicorn
```

**Procfile** (optional):
```
web: uvicorn battlesnake_smart_serpent:app --host 0.0.0.0 --port $PORT
```

### Railway / Fly.io
Use the same command as above. Ensure the exposed port matches their platform setting.

### Docker
**Dockerfile** (minimal):
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY battlesnake_smart_serpent.py ./
RUN pip install --no-cache-dir fastapi uvicorn
ENV PORT=8000
CMD ["uvicorn", "battlesnake_smart_serpent:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build & run:
```bash
docker build -t smart-serpent .
docker run -p 8000:8000 -e AGGRESSION=1.4 smart-serpent
```

---

## Customization
- **Color & cosmetics:** Edit the `GET /` info block.
- **Personality:** Tune `AGGRESSION`; increase/decrease the flood‑fill weight; adjust food threshold.
- **Mode switches:** You can branch behavior by board size, snake count, or late‑game turns.

---

## Troubleshooting
- _The site can’t reach my snake_: ensure your URL is public (ngrok/cloud). Check that the path (`/move`) responds.
- _Latency timeouts_: prefer regions close to Battlesnake servers; keep logs concise; avoid heavy per‑turn I/O.
- _Random deaths near tails_: confirm the tail‑pass logic; only treat tail as free when **not** eating this turn.

---

## License
You are free to use and modify this code for the purposes of the challenge and personal experimentation. If you redistribute, please include attribution to this project.

---

## Acknowledgments
Thanks to the Battlesnake community and docs for the open, well‑specified API that makes building competitive agents fun.

