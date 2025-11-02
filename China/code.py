"""
Smart Battlesnake (FastAPI)

Goals:
- Avoid walls, bodies, and dangerous head-to-heads
- Prefer moves that keep the largest reachable space (flood-fill area heuristic)
- Go for food when health is low or we need to grow vs. opponents
- Opportunistically take winning head-to-heads when we're longer

Quick start locally:
1) pip install fastapi uvicorn
2) python battlesnake_smart_serpent.py
3) Expose locally (e.g., with `ngrok http 8000`) or deploy to Render/railway.
4) In Battlesnake dashboard, set your snake URL to the public endpoint base (e.g., https://<your-app>.onrender.com/)

Environment variables (optional):
- PORT: server port (default 8000)
- LOG_LEVEL: DEBUG/INFO/WARN/ERROR (default INFO)
- AGGRESSION: float in [0,2], >1 makes riskier head-to-heads (default 1.0)

API: Compatible with Battlesnake API v1.
"""

from __future__ import annotations
import os
import math
import random
from collections import deque
from typing import Dict, List, Tuple, Set, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

# -------------------------------
# Config
# -------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
AGGRESSION = float(os.getenv("AGGRESSION", "1.0"))

MOVES = ["up", "down", "left", "right"]
DIRS = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI()

# -------------------------------
# Pydantic models (for clarity; Battlesnake is flexible)
# -------------------------------
class Coord(BaseModel):
    x: int
    y: int

class Snake(BaseModel):
    id: str
    name: str
    health: int
    body: List[Coord]
    head: Coord
    length: int

class Board(BaseModel):
    height: int
    width: int
    food: List[Coord] = []
    hazards: List[Coord] = []
    snakes: List[Snake]

class Game(BaseModel):
    id: str
    timeout: int

class GameState(BaseModel):
    game: Game
    turn: int
    board: Board
    you: Snake

# -------------------------------
# Helpers
# -------------------------------

def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)


def in_bounds(p: Tuple[int, int], w: int, h: int) -> bool:
    x, y = p
    return 0 <= x < w and 0 <= y < h


def add(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def to_tuple(c: Coord) -> Tuple[int,int]:
    return (c.x, c.y)


def bodies_and_tails(board: Board, me_id: str) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]], List[Tuple[int,int]], Dict[str,int]]:
    """Returns (all_body_cells, my_body_cells, opponent_heads, lengths_by_snake_id)."""
    all_bodies: Set[Tuple[int,int]] = set()
    my_body: Set[Tuple[int,int]] = set()
    opp_heads: List[Tuple[int,int]] = []
    lengths: Dict[str,int] = {}

    for s in board.snakes:
        length = s.length
        lengths[s.id] = length
        parts = [to_tuple(p) for p in s.body]
        for p in parts:
            all_bodies.add(p)
        if s.id == me_id:
            for p in parts:
                my_body.add(p)
        else:
            opp_heads.append(to_tuple(s.head))
    return all_bodies, my_body, opp_heads, lengths


def will_eat(new_head: Tuple[int,int], food: Set[Tuple[int,int]]) -> bool:
    return new_head in food


def head_to_head_danger(new_head: Tuple[int,int], opp_heads: List[Tuple[int,int]], lengths: Dict[str,int], my_len: int) -> bool:
    """True if an equal/longer opponent could move into new_head next turn."""
    for oh in opp_heads:
        if manhattan(new_head, oh) == 1 and lengths:  # they could move here
            # length lookup by position needs id; approximate: if any opponent length >= mine, treat as danger
            # We don't have id by head pos here; conservative approach: assume some equal/longer exists nearby
            # Improved: pass also a map from head position to length
            pass
    return False


def head_to_head_score(new_head: Tuple[int,int], opp_heads: List[Tuple[int,int]], opp_head_lengths: Dict[Tuple[int,int],int], my_len: int) -> float:
    """Positive if we are favored in potential head-to-head, negative if we are unfavored."""
    score = 0.0
    for oh in opp_heads:
        if manhattan(new_head, oh) == 1:
            opp_len = opp_head_lengths.get(oh, my_len)  # default conservative
            if my_len > opp_len:
                score += 6.0 * AGGRESSION  # we can bully them
            elif my_len == opp_len:
                score -= 4.0 / AGGRESSION  # risky tie
            else:
                score -= 10.0 / max(AGGRESSION, 1e-6)  # avoid longer heads
    return score


def flood_fill_area(start: Tuple[int,int], blocked: Set[Tuple[int,int]], w: int, h: int, limit: int = 200) -> int:
    if start in blocked or not in_bounds(start, w, h):
        return 0
    q = deque([start])
    seen = {start}
    area = 0
    while q and area < limit:
        p = q.popleft()
        area += 1
        for dx, dy in DIRS.values():
            np = (p[0]+dx, p[1]+dy)
            if np in seen:
                continue
            if not in_bounds(np, w, h):
                continue
            if np in blocked:
                continue
            seen.add(np)
            q.append(np)
    return area


def shortest_path_len(starts: List[Tuple[int,int]], targets: Set[Tuple[int,int]], blocked: Set[Tuple[int,int]], w: int, h: int) -> Optional[int]:
    if not targets:
        return None
    q = deque()
    seen: Set[Tuple[int,int]] = set()
    for s in starts:
        if s in blocked:
            continue
        if not in_bounds(s, w, h):
            continue
        q.append((s,0))
        seen.add(s)
    while q:
        p,d = q.popleft()
        if p in targets:
            return d
        for dx, dy in DIRS.values():
            np = (p[0]+dx, p[1]+dy)
            if np in seen or np in blocked or not in_bounds(np, w, h):
                continue
            seen.add(np)
            q.append((np,d+1))
    return None


def build_head_length_map(board: Board) -> Dict[Tuple[int,int], int]:
    m: Dict[Tuple[int,int],int] = {}
    for s in board.snakes:
        m[to_tuple(s.head)] = s.length
    return m

# -------------------------------
# Core Move Logic
# -------------------------------

def choose_move(gs: GameState) -> str:
    w, h = gs.board.width, gs.board.height
    my = gs.you
    my_head = to_tuple(my.head)
    my_len = my.length
    my_health = my.health

    food_set: Set[Tuple[int,int]] = {to_tuple(f) for f in gs.board.food}
    hazard_set: Set[Tuple[int,int]] = {to_tuple(z) for z in gs.board.hazards}

    all_bodies, my_body, opp_heads, lengths = bodies_and_tails(gs.board, my.id)
    opp_head_len_map = build_head_length_map(gs.board)

    # Consider own tail as passable if we won't eat this turn (tail moves)
    my_tail = to_tuple(my.body[-1]) if my.body else None

    candidates: List[Tuple[str, Tuple[int,int]]] = []
    for mv, (dx,dy) in DIRS.items():
        nh = (my_head[0] + dx, my_head[1] + dy)
        candidates.append((mv, nh))

    # Score each move
    best_score = -1e18
    best_moves: List[str] = []

    for mv, nh in candidates:
        # Hard constraints
        if not in_bounds(nh, w, h):
            continue

        # Determine if tail is safe to occupy for us
        eating = will_eat(nh, food_set)
        blocked = set(all_bodies)
        if my_tail is not None and not eating:
            # Tail will move, so stepping onto current tail is okay
            blocked.discard(my_tail)

        if nh in blocked:
            continue

        # Base score
        score = 0.0

        # Hazard penalty (Royale); prefer non-hazard
        if nh in hazard_set:
            score -= 8.0

        # Distance to board center as a mild prior (avoid hugging walls forever)
        cx, cy = (w-1)/2.0, (h-1)/2.0
        dist_center = math.hypot(nh[0]-cx, nh[1]-cy)
        score += -0.05 * dist_center

        # Avoid moving next to longer or equal heads (possible head-to-head ties/losses)
        score += head_to_head_score(nh, opp_heads, opp_head_len_map, my_len)

        # Space heuristic: how many cells reachable if we move here?
        area = flood_fill_area(nh, blocked, w, h, limit=400)
        score += 0.2 * area  # space is life

        # Food heuristic: dynamic target
        need_food = (my_health <= 40) or (my_len <= max([s.length for s in gs.board.snakes] + [my_len]) - 2)
        if need_food and food_set:
            # Estimate shortest path to any food from this next head
            d_food = shortest_path_len([nh], food_set, blocked, w, h)
            if d_food is not None:
                score += 8.0 / (1.0 + d_food)
            # Bonus if moving directly onto food now
            if nh in food_set:
                score += 6.0
        else:
            # If we're healthy and big, being in open space is more important
            score += 0.05 * area

        # Gentle preference to continue current heading to reduce jitter
        # (Infer previous direction if body length>1)
        if len(my.body) >= 2:
            neck = to_tuple(my.body[1])
            prev_dir = (my_head[0]-neck[0], my_head[1]-neck[1])
            if DIRS.get(mv) == prev_dir:
                score += 0.2

        # Tiebreak: random small noise for diversity
        score += random.uniform(-0.05, 0.05)

        if score > best_score:
            best_score = score
            best_moves = [mv]
        elif abs(score - best_score) < 1e-6:
            best_moves.append(mv)

    if not best_moves:
        # No legal moves; pick any to satisfy API, likely game over
        return random.choice(MOVES)

    # If multiple best, prefer move that increases distance from nearest longer head
    longer_heads = [pos for pos, L in opp_head_len_map.items() if L >= my_len]
    if longer_heads:
        def safety_metric(mv: str) -> float:
            nh = (my_head[0] + DIRS[mv][0], my_head[1] + DIRS[mv][1])
            return min(manhattan(nh, lh) for lh in longer_heads)
        best_moves.sort(key=safety_metric, reverse=True)

    return best_moves[0]

# -------------------------------
# Battlesnake API endpoints
# -------------------------------
@app.get("/")
def get_info() -> Dict:
    log("INFO")
    return {
        "apiversion": "1",
        "author": "smart-serpent",
        "color": "#22c55e",
        "head": "beluga",
        "tail": "bolt",
    }

@app.post("/start")
async def on_start(req: Request) -> Dict:
    gs = GameState(**(await req.json()))
    log(f"GAME START: {gs.game.id}")
    return {"color": "#22c55e"}

@app.post("/move")
async def on_move(req: Request) -> Dict:
    gs = GameState(**(await req.json()))
    mv = choose_move(gs)
    shout = f"t{gs.turn}:{mv}"
    log(f"TURN {gs.turn} -> {mv}")
    return {"move": mv, "shout": shout}

@app.post("/end")
async def on_end(req: Request) -> Dict:
    gs = GameState(**(await req.json()))
    log(f"GAME OVER: {gs.game.id}")
    return {}

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
