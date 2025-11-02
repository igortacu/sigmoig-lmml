# Arcade Olympics Bot — How It Works

This bot plays four mini‑games **simultaneously** but outputs a **single** action (`UP/RIGHT/DOWN/LEFT`) each turn. It builds a per‑action score **inside each mini‑game**, then either **gates** the decision to the most decisive game(s) (critical phases) or **fuses** all games with weights. It is fully deterministic and designed to avoid late‑round collapses.

---

## Scoring Objective (Global)
Final tournament score is the **product** of per‑game scores, with each mini‑game score computed as:
```
mini_game_score = nb_silver + 3 * nb_gold
```
Bronze contributes 0 to this formula. Because of the product, the bot **upweights any mini‑game** where we still have **no silver/gold** (“zero‑product guard”).

---

## Data Used Each Turn
- **`scoreInfo` (ours)**: 13 integers per spec → extract `(g_i, s_i)` to maintain **per‑game medal points** `points[i] = s_i + 3*g_i`.
- **Per‑game state** (`GPU`, `reg0..reg6`): interpreted by each mini‑game module.
- **Persistent state**: `DivingTracker(goal, idx)` to align with the current diving sequence across turns; resets on `GPU == "GAME_OVER"`.

---

## Per‑Game Heuristics (Per‑Action Scores)

### 1) Hurdle Race
- **Simulation** from current position `pos`:
  - `UP`: jump to `pos+2`, but **landing** can still collide with a hurdle.
  - `LEFT/DOWN/RIGHT`: move 1/2/3 steps, **stopping on a hurdle** if encountered.
- **Score =** `+1.2 × progress` `+10 if finish` `+0.35 × overtakes` `−9 if collide`  
  *Rationale:* finishing and avoiding the 3‑turn stun are paramount; small reward for overtaking.
- If stunned (`reg3+me > 0`), all actions are scored neutral (no movement).

### 2) Archery
- `GPU[0]` is current **wind strength** `s`. Greedy one‑step move to **reduce Euclidean distance** to `(0,0)`, then clamp to `[-20, 20]`.  
- Direction semantics: `UP` → `y -= s`, `DOWN` → `y += s`, `LEFT/RIGHT` → `x ±= s`.
- **Score =** `−sqrt(x'^2 + y'^2)` (closer is better).

### 3) Roller Speed Skating
- `GPU` is a permutation (risk order) of `U L D R` → maps to step/risk:
  - index 0 → `+1 step, risk −1`
  - index 1 → `+2 steps`
  - index 2 → `+2 steps, risk +1`
  - index 3 → `+3 steps, risk +2`
- **Collision** (same tile modulo 10 as an opponent) adds `+2 risk`. `risk ≥ 5` ⇒ **stunned 2 turns** next.
- **Score =** `+1.35 × steps` `− (risk_penalty) × max(0, risk'−2) × 0.6` `−1.0 if collide` `− (stun_penalty) if risk' ≥ 5`  
  with **adaptive penalties**: as `turns_left` decreases, we tolerate more risk (smaller `risk_penalty`), but still avoid a fatal stun (`stun_penalty` large). Minor **lead bonus** in last 2 turns for being ahead modulo 10.
- If currently stunned (`risk < 0`), actions are neutral.

### 4) Diving
- Persist the **goal sequence** (e.g., `UUUDDLL...`) and an **index** into it across the run.
- If action matches the current symbol, immediate gain is approximated as `combo+1`; otherwise 0.  
- Resets index on `GPU == "GAME_OVER"` (between runs).

---

## Critical‑Phase Gating

The agent detects decisive moments and **lets the decisive game dominate** that turn. A game is *critical* when:

- **Hurdles:** anyone ≤ 3 tiles from finish, or we ≤ 4, or a hurdle is directly ahead.
- **Archery:** very short remaining wind (`len(GPU) ≤ 2`) **or** we’re still far from center (`|x|+|y| ≥ 16`).
- **Skating:** `turns_left ≤ 3` **or** `risk ≥ 4` (near‑stun boundary).
- **Diving:** `combo ≥ 4` **or** ≤ 2 symbols left in the current sequence.

If **any** game is critical, only those games’ normalized scores are aggregated:
```
action_score = Σ_{critical games} ( 2.2 × weight(game) × normalized_score_game[action] )
```
Otherwise, a smooth fusion is used:
```
action_score = Σ_{all games} ( weight(game) × normalized_score_game[action] )
```

---

## Normalization & Weights

- **Per‑game min‑max normalization** maps raw action scores to `[0,1]` to make games comparable; flat maps yield zeros.
- **Base weights** (Hurdles, Archery, Skating, Diving) = `[1.1, 0.9, 1.1, 1.0]` (slight preference to racing).
- **Zero‑product guard:** if `points[i] == 0` (no S/G medals yet in that mini‑game), add `+0.7` to `weight(i)` until some S/G is secured.

---

## Decision & Tie‑Break
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcade Olympics multi-game agent — Critical-Phase Gating Strategy

Key upgrade vs baseline:
- Detects CRITICAL phases per mini-game and *gates* the decision to the decisive game(s)
  (Hurdles near finish, Skating with few turns or high risk, Diving with big combo or near sequence end,
   Archery in last winds). Otherwise uses weighted fusion.
- Safer hurdles sim, smarter skating risk, stronger late-game focus (the usual place you lose).

Assumed game order (per league statement): 0 Hurdles, 1 Archery, 2 Skating, 3 Diving.
"""

import sys
import math

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRS = {"U":"UP","R":"RIGHT","D":"DOWN","L":"LEFT"}
INV = {"UP":"U","RIGHT":"R","DOWN":"D","LEFT":"L"}

# ---------- Helpers ----------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def norm_scores(score_map):
    vals = list(score_map.values())
    if not vals:
        return {a: 0.0 for a in ACTIONS}
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {a: 0.0 for a in ACTIONS}
    return {a: (score_map[a] - mn) / (mx - mn) for a in ACTIONS}

def ints_from_line(line):
    return list(map(int, line.strip().split()))

# ---------- Persistent state ----------

class DivingTracker:
    def __init__(self):
        self.goal = ""
        self.idx = 0
        self.running = False

    def reset(self):
        self.idx = 0
        self.running = False

    def start_or_sync(self, goal):
        if goal != self.goal:
            self.goal = goal
            self.idx = 0
        self.running = True

    def advance(self):
        if self.goal:
            self.idx = (self.idx + 1) % len(self.goal)

class Agent:
    def __init__(self, me):
        self.me = me
        self.nb_games = 4
        self.turn = 0

        self.G_HURD = 0
        self.G_ARCH = 1
        self.G_SKAT = 2
        self.G_DIVE = 3

        self.dive = DivingTracker()

        # Track per-game medal "points" = silver + 3*gold (bronze doesn't count)
        self.mini_points = [0, 0, 0, 0]

    # ---------- Score parsing -> medal points ----------
    def update_medal_points(self, my_score_line):
        # 13 ints: [final_score, g1 s1 b1, g2 s2 b2, g3 s3 b3, g4 s4 b4]
        xs = ints_from_line(my_score_line)
        if len(xs) >= 13:
            _, g1, s1, _, g2, s2, _, g3, s3, _, g4, s4, _ = xs[:13]
            self.mini_points = [s1 + 3*g1, s2 + 3*g2, s3 + 3*g3, s4 + 3*g4]

    # ---------- Game scorers ----------
    def score_hurdles(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":  # reset frame
            return scores

        track = gpu
        n = len(track)
        fin = n - 1
        pos = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0:
            return scores  # can’t move this turn

        others = [regs[0], regs[1], regs[2]]

        def is_hurdle(i):
            if i < 0 or i >= n: return False
            return track[i] == '#'

        def simulate(action):
            if action == "UP":
                # Jump over next space, land 2 ahead; hurdle on landing still hurts
                tgt = min(pos + 2, fin)
                collided = is_hurdle(tgt)  # only landing matters for UP
                return tgt, collided, (tgt >= fin)

            steps = {"LEFT":1, "DOWN":2, "RIGHT":3}[action]
            cur = pos
            collided = False
            for _ in range(steps):
                nxt = min(cur + 1, fin)
                if is_hurdle(nxt):
                    cur = nxt
                    collided = True
                    break
                cur = nxt
            return cur, collided, (cur >= fin)

        # Heavier penalty for collision (3-turn stun)
        for a in ACTIONS:
            newp, collided, finished = simulate(a)
            gain = newp - pos
            score = 0.0
            if finished: score += 10.0
            score += 1.2 * gain
            if collided: score -= 9.0
            # prefer overtakes
            score += 0.35 * sum(1 for op in others if newp > op)
            # tiny bias to moves that keep us aligned for an UP jump over upcoming hurdles
            if a != "UP" and not collided:
                if is_hurdle(min(newp+1, fin)) and not is_hurdle(min(newp+2, fin)):
                    score += 0.2
            scores[a] = score
        return scores

    def score_archery(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or not gpu:
            return scores

        base = 2 * self.me
        x, y = regs[base], regs[base + 1]

        # current wind strength is the first integer (char digit in examples)
        try:
            s = int(gpu[0])
        except:
            # if multi-digit encoding exists, fall back to first char digit
            s = 0

        def dist2(xx, yy):
            return xx*xx + yy*yy

        # One-step greedy (cap within [-20,20])
        for a in ACTIONS:
            nx, ny = x, y
            if a == "LEFT":  nx -= s
            if a == "RIGHT": nx += s
            if a == "UP":    ny -= s  # define UP as negative y
            if a == "DOWN":  ny += s
            nx = clamp(nx, -20, 20)
            ny = clamp(ny, -20, 20)
            scores[a] = -math.sqrt(dist2(nx, ny))
        return scores

    def score_skating(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or len(gpu) != 4:
            return scores

        order = gpu  # e.g., "ULDR"
        pos = regs[0 + self.me]
        risk = regs[3 + self.me]
        stunned = (risk < 0)
        turns_left = max(0, regs[6])

        opp_pos = [regs[0], regs[1], regs[2]]

        if stunned:
            return scores

        # Calibrate penalties: late game -> allow more risk to secure place
        endgame_factor = 1.0 - min(1.0, turns_left / 12.0)  # -> 1 when very near end
        risk_penalty = 0.5 + 0.4 * (1.0 - endgame_factor)   # smaller near end
        stun_penalty = 7.0 + 3.0 * (1.0 - endgame_factor)   # still avoid stuns

        def eval_action(a):
            try:
                idx = order.index(INV[a])
            except ValueError:
                return -1e9

            if idx == 0: step, dr = 1, -1
            elif idx == 1: step, dr = 2, 0
            elif idx == 2: step, dr = 2, +1
            else:          step, dr = 3, +2

            npos = pos + step
            nrisk = max(0, risk + dr)

            # collision on same tile modulo 10
            collide = any(((npos % 10) == (op % 10)) for i, op in enumerate(opp_pos) if i != self.me)
            if collide:
                nrisk += 2

            will_stun = (nrisk >= 5)

            score = 0.0
            score += 1.35 * step
            if collide:   score -= 1.0
            score -= risk_penalty * max(0, nrisk - 2) * 0.6
            if will_stun: score -= stun_penalty

            # Micro edge: if endgame, reward being ahead on lap modulo 10
            if turns_left <= 2:
                lead = sum(1 for op in opp_pos if (npos % 10) > (op % 10))
                score += 0.3 * lead
            return score

        for a in ACTIONS:
            scores[a] = eval_action(a)
        return scores

    def score_diving(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":
            self.dive.reset()
            return scores

        self.dive.start_or_sync(gpu)
        if not self.dive.goal:
            return scores

        combo = regs[3 + self.me]
        want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
        want_action = DIRS.get(want, None)

        for a in ACTIONS:
            if a == want_action:
                scores[a] = (combo + 1) * 1.0
            else:
                scores[a] = 0.0
        return scores

    # ---------- Critical gating logic ----------
    def is_hurdles_critical(self, gpu, regs):
        if gpu == "GAME_OVER": return False
        track = gpu
        fin = len(track) - 1
        pos_me = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0: return False
        dist_me = fin - pos_me
        dist_any = min(fin - regs[0], fin - regs[1], fin - regs[2])
        # Critical if someone <=3 from finish, or we <=4 (one push away), or a hurdle right in front
        hurdle_next = (pos_me + 1 <= fin and track[pos_me + 1] == '#')
        return (dist_any <= 3) or (dist_me <= 4) or hurdle_next

    def is_archery_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        base = 2 * self.me
        x, y = regs[base], regs[base+1]
        dist = abs(x) + abs(y)
        # If only 1-2 winds are visible OR we are still far from the center, push harder
        return (len(gpu) <= 2) or (dist >= 16)

    def is_skating_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or len(gpu) != 4: return False
        risk = regs[3 + self.me]
        turns_left = max(0, regs[6])
        # Near end or near stun boundary is critical
        return (turns_left <= 3) or (risk >= 4)

    def is_diving_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        combo = regs[3 + self.me]
        if not self.dive.goal:
            return False
        steps_left_in_seq = (len(self.dive.goal) - (self.dive.idx % len(self.dive.goal))) % len(self.dive.goal)
        # Critical if big combo (high marginal points) or near end of sequence
        return (combo >= 4) or (steps_left_in_seq <= 2)

    def weight_for_game(self, gi):
        # Base weights with small preference for racing games
        base = [1.1, 0.9, 1.1, 1.0][gi]
        # If points for that game are still zero (no S/G), push harder to avoid zero product
        if gi < len(self.mini_points) and self.mini_points[gi] == 0:
            base += 0.7
        return base

    # ---------- Decision ----------
    def decide(self, games, score_lines):
        self.turn += 1
        self.update_medal_points(score_lines[self.me])

        # Per-game raw scores
        s_h = self.score_hurdles(games[self.G_HURD][0], games[self.G_HURD][1])
        s_a = self.score_archery(games[self.G_ARCH][0], games[self.G_ARCH][1])
        s_s = self.score_skating(games[self.G_SKAT][0], games[self.G_SKAT][1])
        s_d = self.score_diving(games[self.G_DIVE][0], games[self.G_DIVE][1])

        # Normalize
        n_h, n_a, n_s, n_d = map(norm_scores, (s_h, s_a, s_s, s_d))

        # Critical gating
        crit = [
            self.is_hurdles_critical(games[self.G_HURD][0], games[self.G_HURD][1]),
            self.is_archery_critical(games[self.G_ARCH][0], games[self.G_ARCH][1]),
            self.is_skating_critical(games[self.G_SKAT][0], games[self.G_SKAT][1]),
            self.is_diving_critical(games[self.G_DIVE][0], games[self.G_DIVE][1]),
        ]

        # If any game is critical, emphasize only those (winner-take-most)
        action_scores = {a: 0.0 for a in ACTIONS}
        if any(crit):
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                if crit[gi]:
                    w = 2.2 * self.weight_for_game(gi)   # upweight critical ones
                    for a in ACTIONS:
                        action_scores[a] += w * nmap.get(a, 0.0)
        else:
            # Smooth fusion when nothing is urgent
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                w = self.weight_for_game(gi)
                for a in ACTIONS:
                    action_scores[a] += w * nmap.get(a, 0.0)

        # Final tiebreak: favor the action matching current Diving symbol (keeps combo snowballing)
        dive_bias = 0.0
        want_act = None
        if self.dive.goal:
            want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
            want_act = DIRS.get(want, None)

        best = max(
            ACTIONS,
            key=lambda a: (action_scores[a],
                           0.45 if (want_act and a == want_act) else 0.0,
                           -ACTIONS.index(a))
        )

        # Advance diving index to stay aligned with the run
        if self.dive.running and self.dive.goal:
            self.dive.advance()

        return best

# ---------- I/O loop ----------

def main():
    rd = sys.stdin
    me = int(rd.readline().strip())
    nb = int(rd.readline().strip())

    agent = Agent(me)
    agent.nb_games = nb

    while True:
        # 3 score lines (players 0..2)
        score_lines = []
        for _ in range(3):
            line = rd.readline()
            if not line:
                return
            score_lines.append(line.rstrip("\n"))

        # nb game lines
        games = []
        for _ in range(nb):
            line = rd.readline()
            if not line:
                return
            parts = line.strip().split()
            gpu = parts[0]
            regs = list(map(int, parts[1:8])) if len(parts) >= 8 else [-1]*7
            games.append((gpu, regs))

        act = agent.decide(games, score_lines)
        print(act, flush=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcade Olympics multi-game agent — Critical-Phase Gating Strategy

Key upgrade vs baseline:
- Detects CRITICAL phases per mini-game and *gates* the decision to the decisive game(s)
  (Hurdles near finish, Skating with few turns or high risk, Diving with big combo or near sequence end,
   Archery in last winds). Otherwise uses weighted fusion.
- Safer hurdles sim, smarter skating risk, stronger late-game focus (the usual place you lose).

Assumed game order (per league statement): 0 Hurdles, 1 Archery, 2 Skating, 3 Diving.
"""

import sys
import math

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRS = {"U":"UP","R":"RIGHT","D":"DOWN","L":"LEFT"}
INV = {"UP":"U","RIGHT":"R","DOWN":"D","LEFT":"L"}

# ---------- Helpers ----------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def norm_scores(score_map):
    vals = list(score_map.values())
    if not vals:
        return {a: 0.0 for a in ACTIONS}
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {a: 0.0 for a in ACTIONS}
    return {a: (score_map[a] - mn) / (mx - mn) for a in ACTIONS}

def ints_from_line(line):
    return list(map(int, line.strip().split()))

# ---------- Persistent state ----------

class DivingTracker:
    def __init__(self):
        self.goal = ""
        self.idx = 0
        self.running = False

    def reset(self):
        self.idx = 0
        self.running = False

    def start_or_sync(self, goal):
        if goal != self.goal:
            self.goal = goal
            self.idx = 0
        self.running = True

    def advance(self):
        if self.goal:
            self.idx = (self.idx + 1) % len(self.goal)

class Agent:
    def __init__(self, me):
        self.me = me
        self.nb_games = 4
        self.turn = 0

        self.G_HURD = 0
        self.G_ARCH = 1
        self.G_SKAT = 2
        self.G_DIVE = 3

        self.dive = DivingTracker()

        # Track per-game medal "points" = silver + 3*gold (bronze doesn't count)
        self.mini_points = [0, 0, 0, 0]

    # ---------- Score parsing -> medal points ----------
    def update_medal_points(self, my_score_line):
        # 13 ints: [final_score, g1 s1 b1, g2 s2 b2, g3 s3 b3, g4 s4 b4]
        xs = ints_from_line(my_score_line)
        if len(xs) >= 13:
            _, g1, s1, _, g2, s2, _, g3, s3, _, g4, s4, _ = xs[:13]
            self.mini_points = [s1 + 3*g1, s2 + 3*g2, s3 + 3*g3, s4 + 3*g4]

    # ---------- Game scorers ----------
    def score_hurdles(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":  # reset frame
            return scores

        track = gpu
        n = len(track)
        fin = n - 1
        pos = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0:
            return scores  # can’t move this turn

        others = [regs[0], regs[1], regs[2]]

        def is_hurdle(i):
            if i < 0 or i >= n: return False
            return track[i] == '#'

        def simulate(action):
            if action == "UP":
                # Jump over next space, land 2 ahead; hurdle on landing still hurts
                tgt = min(pos + 2, fin)
                collided = is_hurdle(tgt)  # only landing matters for UP
                return tgt, collided, (tgt >= fin)

            steps = {"LEFT":1, "DOWN":2, "RIGHT":3}[action]
            cur = pos
            collided = False
            for _ in range(steps):
                nxt = min(cur + 1, fin)
                if is_hurdle(nxt):
                    cur = nxt
                    collided = True
                    break
                cur = nxt
            return cur, collided, (cur >= fin)

        # Heavier penalty for collision (3-turn stun)
        for a in ACTIONS:
            newp, collided, finished = simulate(a)
            gain = newp - pos
            score = 0.0
            if finished: score += 10.0
            score += 1.2 * gain
            if collided: score -= 9.0
            # prefer overtakes
            score += 0.35 * sum(1 for op in others if newp > op)
            # tiny bias to moves that keep us aligned for an UP jump over upcoming hurdles
            if a != "UP" and not collided:
                if is_hurdle(min(newp+1, fin)) and not is_hurdle(min(newp+2, fin)):
                    score += 0.2
            scores[a] = score
        return scores

    def score_archery(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or not gpu:
            return scores

        base = 2 * self.me
        x, y = regs[base], regs[base + 1]

        # current wind strength is the first integer (char digit in examples)
        try:
            s = int(gpu[0])
        except:
            # if multi-digit encoding exists, fall back to first char digit
            s = 0

        def dist2(xx, yy):
            return xx*xx + yy*yy

        # One-step greedy (cap within [-20,20])
        for a in ACTIONS:
            nx, ny = x, y
            if a == "LEFT":  nx -= s
            if a == "RIGHT": nx += s
            if a == "UP":    ny -= s  # define UP as negative y
            if a == "DOWN":  ny += s
            nx = clamp(nx, -20, 20)
            ny = clamp(ny, -20, 20)
            scores[a] = -math.sqrt(dist2(nx, ny))
        return scores

    def score_skating(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or len(gpu) != 4:
            return scores

        order = gpu  # e.g., "ULDR"
        pos = regs[0 + self.me]
        risk = regs[3 + self.me]
        stunned = (risk < 0)
        turns_left = max(0, regs[6])

        opp_pos = [regs[0], regs[1], regs[2]]

        if stunned:
            return scores

        # Calibrate penalties: late game -> allow more risk to secure place
        endgame_factor = 1.0 - min(1.0, turns_left / 12.0)  # -> 1 when very near end
        risk_penalty = 0.5 + 0.4 * (1.0 - endgame_factor)   # smaller near end
        stun_penalty = 7.0 + 3.0 * (1.0 - endgame_factor)   # still avoid stuns

        def eval_action(a):
            try:
                idx = order.index(INV[a])
            except ValueError:
                return -1e9

            if idx == 0: step, dr = 1, -1
            elif idx == 1: step, dr = 2, 0
            elif idx == 2: step, dr = 2, +1
            else:          step, dr = 3, +2

            npos = pos + step
            nrisk = max(0, risk + dr)

            # collision on same tile modulo 10
            collide = any(((npos % 10) == (op % 10)) for i, op in enumerate(opp_pos) if i != self.me)
            if collide:
                nrisk += 2

            will_stun = (nrisk >= 5)

            score = 0.0
            score += 1.35 * step
            if collide:   score -= 1.0
            score -= risk_penalty * max(0, nrisk - 2) * 0.6
            if will_stun: score -= stun_penalty

            # Micro edge: if endgame, reward being ahead on lap modulo 10
            if turns_left <= 2:
                lead = sum(1 for op in opp_pos if (npos % 10) > (op % 10))
                score += 0.3 * lead
            return score

        for a in ACTIONS:
            scores[a] = eval_action(a)
        return scores

    def score_diving(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":
            self.dive.reset()
            return scores

        self.dive.start_or_sync(gpu)
        if not self.dive.goal:
            return scores

        combo = regs[3 + self.me]
        want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
        want_action = DIRS.get(want, None)

        for a in ACTIONS:
            if a == want_action:
                scores[a] = (combo + 1) * 1.0
            else:
                scores[a] = 0.0
        return scores

    # ---------- Critical gating logic ----------
    def is_hurdles_critical(self, gpu, regs):
        if gpu == "GAME_OVER": return False
        track = gpu
        fin = len(track) - 1
        pos_me = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0: return False
        dist_me = fin - pos_me
        dist_any = min(fin - regs[0], fin - regs[1], fin - regs[2])
        # Critical if someone <=3 from finish, or we <=4 (one push away), or a hurdle right in front
        hurdle_next = (pos_me + 1 <= fin and track[pos_me + 1] == '#')
        return (dist_any <= 3) or (dist_me <= 4) or hurdle_next

    def is_archery_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        base = 2 * self.me
        x, y = regs[base], regs[base+1]
        dist = abs(x) + abs(y)
        # If only 1-2 winds are visible OR we are still far from the center, push harder
        return (len(gpu) <= 2) or (dist >= 16)

    def is_skating_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or len(gpu) != 4: return False
        risk = regs[3 + self.me]
        turns_left = max(0, regs[6])
        # Near end or near stun boundary is critical
        return (turns_left <= 3) or (risk >= 4)

    def is_diving_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        combo = regs[3 + self.me]
        if not self.dive.goal:
            return False
        steps_left_in_seq = (len(self.dive.goal) - (self.dive.idx % len(self.dive.goal))) % len(self.dive.goal)
        # Critical if big combo (high marginal points) or near end of sequence
        return (combo >= 4) or (steps_left_in_seq <= 2)

    def weight_for_game(self, gi):
        # Base weights with small preference for racing games
        base = [1.1, 0.9, 1.1, 1.0][gi]
        # If points for that game are still zero (no S/G), push harder to avoid zero product
        if gi < len(self.mini_points) and self.mini_points[gi] == 0:
            base += 0.7
        return base

    # ---------- Decision ----------
    def decide(self, games, score_lines):
        self.turn += 1
        self.update_medal_points(score_lines[self.me])

        # Per-game raw scores
        s_h = self.score_hurdles(games[self.G_HURD][0], games[self.G_HURD][1])
        s_a = self.score_archery(games[self.G_ARCH][0], games[self.G_ARCH][1])
        s_s = self.score_skating(games[self.G_SKAT][0], games[self.G_SKAT][1])
        s_d = self.score_diving(games[self.G_DIVE][0], games[self.G_DIVE][1])

        # Normalize
        n_h, n_a, n_s, n_d = map(norm_scores, (s_h, s_a, s_s, s_d))

        # Critical gating
        crit = [
            self.is_hurdles_critical(games[self.G_HURD][0], games[self.G_HURD][1]),
            self.is_archery_critical(games[self.G_ARCH][0], games[self.G_ARCH][1]),
            self.is_skating_critical(games[self.G_SKAT][0], games[self.G_SKAT][1]),
            self.is_diving_critical(games[self.G_DIVE][0], games[self.G_DIVE][1]),
        ]

        # If any game is critical, emphasize only those (winner-take-most)
        action_scores = {a: 0.0 for a in ACTIONS}
        if any(crit):
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                if crit[gi]:
                    w = 2.2 * self.weight_for_game(gi)   # upweight critical ones
                    for a in ACTIONS:
                        action_scores[a] += w * nmap.get(a, 0.0)
        else:
            # Smooth fusion when nothing is urgent
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                w = self.weight_for_game(gi)
                for a in ACTIONS:
                    action_scores[a] += w * nmap.get(a, 0.0)

        # Final tiebreak: favor the action matching current Diving symbol (keeps combo snowballing)
        dive_bias = 0.0
        want_act = None
        if self.dive.goal:
            want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
            want_act = DIRS.get(want, None)

        best = max(
            ACTIONS,
            key=lambda a: (action_scores[a],
                           0.45 if (want_act and a == want_act) else 0.0,
                           -ACTIONS.index(a))
        )

        # Advance diving index to stay aligned with the run
        if self.dive.running and self.dive.goal:
            self.dive.advance()

        return best

# ---------- I/O loop ----------

def main():
    rd = sys.stdin
    me = int(rd.readline().strip())
    nb = int(rd.readline().strip())

    agent = Agent(me)
    agent.nb_games = nb

    while True:
        # 3 score lines (players 0..2)
        score_lines = []
        for _ in range(3):
            line = rd.readline()
            if not line:
                return
            score_lines.append(line.rstrip("\n"))

        # nb game lines
        games = []
        for _ in range(nb):
            line = rd.readline()
            if not line:
                return
            parts = line.strip().split()
            gpu = parts[0]
            regs = list(map(int, parts[1:8])) if len(parts) >= 8 else [-1]*7
            games.append((gpu, regs))

        act = agent.decide(games, score_lines)
        print(act, flush=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arcade Olympics multi-game agent — Critical-Phase Gating Strategy

Key upgrade vs baseline:
- Detects CRITICAL phases per mini-game and *gates* the decision to the decisive game(s)
  (Hurdles near finish, Skating with few turns or high risk, Diving with big combo or near sequence end,
   Archery in last winds). Otherwise uses weighted fusion.
- Safer hurdles sim, smarter skating risk, stronger late-game focus (the usual place you lose).

Assumed game order (per league statement): 0 Hurdles, 1 Archery, 2 Skating, 3 Diving.
"""

import sys
import math

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRS = {"U":"UP","R":"RIGHT","D":"DOWN","L":"LEFT"}
INV = {"UP":"U","RIGHT":"R","DOWN":"D","LEFT":"L"}

# ---------- Helpers ----------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def norm_scores(score_map):
    vals = list(score_map.values())
    if not vals:
        return {a: 0.0 for a in ACTIONS}
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {a: 0.0 for a in ACTIONS}
    return {a: (score_map[a] - mn) / (mx - mn) for a in ACTIONS}

def ints_from_line(line):
    return list(map(int, line.strip().split()))

# ---------- Persistent state ----------

class DivingTracker:
    def __init__(self):
        self.goal = ""
        self.idx = 0
        self.running = False

    def reset(self):
        self.idx = 0
        self.running = False

    def start_or_sync(self, goal):
        if goal != self.goal:
            self.goal = goal
            self.idx = 0
        self.running = True

    def advance(self):
        if self.goal:
            self.idx = (self.idx + 1) % len(self.goal)

class Agent:
    def __init__(self, me):
        self.me = me
        self.nb_games = 4
        self.turn = 0

        self.G_HURD = 0
        self.G_ARCH = 1
        self.G_SKAT = 2
        self.G_DIVE = 3

        self.dive = DivingTracker()

        # Track per-game medal "points" = silver + 3*gold (bronze doesn't count)
        self.mini_points = [0, 0, 0, 0]

    # ---------- Score parsing -> medal points ----------
    def update_medal_points(self, my_score_line):
        # 13 ints: [final_score, g1 s1 b1, g2 s2 b2, g3 s3 b3, g4 s4 b4]
        xs = ints_from_line(my_score_line)
        if len(xs) >= 13:
            _, g1, s1, _, g2, s2, _, g3, s3, _, g4, s4, _ = xs[:13]
            self.mini_points = [s1 + 3*g1, s2 + 3*g2, s3 + 3*g3, s4 + 3*g4]

    # ---------- Game scorers ----------
    def score_hurdles(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":  # reset frame
            return scores

        track = gpu
        n = len(track)
        fin = n - 1
        pos = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0:
            return scores  # can’t move this turn

        others = [regs[0], regs[1], regs[2]]

        def is_hurdle(i):
            if i < 0 or i >= n: return False
            return track[i] == '#'

        def simulate(action):
            if action == "UP":
                # Jump over next space, land 2 ahead; hurdle on landing still hurts
                tgt = min(pos + 2, fin)
                collided = is_hurdle(tgt)  # only landing matters for UP
                return tgt, collided, (tgt >= fin)

            steps = {"LEFT":1, "DOWN":2, "RIGHT":3}[action]
            cur = pos
            collided = False
            for _ in range(steps):
                nxt = min(cur + 1, fin)
                if is_hurdle(nxt):
                    cur = nxt
                    collided = True
                    break
                cur = nxt
            return cur, collided, (cur >= fin)

        # Heavier penalty for collision (3-turn stun)
        for a in ACTIONS:
            newp, collided, finished = simulate(a)
            gain = newp - pos
            score = 0.0
            if finished: score += 10.0
            score += 1.2 * gain
            if collided: score -= 9.0
            # prefer overtakes
            score += 0.35 * sum(1 for op in others if newp > op)
            # tiny bias to moves that keep us aligned for an UP jump over upcoming hurdles
            if a != "UP" and not collided:
                if is_hurdle(min(newp+1, fin)) and not is_hurdle(min(newp+2, fin)):
                    score += 0.2
            scores[a] = score
        return scores

    def score_archery(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or not gpu:
            return scores

        base = 2 * self.me
        x, y = regs[base], regs[base + 1]

        # current wind strength is the first integer (char digit in examples)
        try:
            s = int(gpu[0])
        except:
            # if multi-digit encoding exists, fall back to first char digit
            s = 0

        def dist2(xx, yy):
            return xx*xx + yy*yy

        # One-step greedy (cap within [-20,20])
        for a in ACTIONS:
            nx, ny = x, y
            if a == "LEFT":  nx -= s
            if a == "RIGHT": nx += s
            if a == "UP":    ny -= s  # define UP as negative y
            if a == "DOWN":  ny += s
            nx = clamp(nx, -20, 20)
            ny = clamp(ny, -20, 20)
            scores[a] = -math.sqrt(dist2(nx, ny))
        return scores

    def score_skating(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER" or len(gpu) != 4:
            return scores

        order = gpu  # e.g., "ULDR"
        pos = regs[0 + self.me]
        risk = regs[3 + self.me]
        stunned = (risk < 0)
        turns_left = max(0, regs[6])

        opp_pos = [regs[0], regs[1], regs[2]]

        if stunned:
            return scores

        # Calibrate penalties: late game -> allow more risk to secure place
        endgame_factor = 1.0 - min(1.0, turns_left / 12.0)  # -> 1 when very near end
        risk_penalty = 0.5 + 0.4 * (1.0 - endgame_factor)   # smaller near end
        stun_penalty = 7.0 + 3.0 * (1.0 - endgame_factor)   # still avoid stuns

        def eval_action(a):
            try:
                idx = order.index(INV[a])
            except ValueError:
                return -1e9

            if idx == 0: step, dr = 1, -1
            elif idx == 1: step, dr = 2, 0
            elif idx == 2: step, dr = 2, +1
            else:          step, dr = 3, +2

            npos = pos + step
            nrisk = max(0, risk + dr)

            # collision on same tile modulo 10
            collide = any(((npos % 10) == (op % 10)) for i, op in enumerate(opp_pos) if i != self.me)
            if collide:
                nrisk += 2

            will_stun = (nrisk >= 5)

            score = 0.0
            score += 1.35 * step
            if collide:   score -= 1.0
            score -= risk_penalty * max(0, nrisk - 2) * 0.6
            if will_stun: score -= stun_penalty

            # Micro edge: if endgame, reward being ahead on lap modulo 10
            if turns_left <= 2:
                lead = sum(1 for op in opp_pos if (npos % 10) > (op % 10))
                score += 0.3 * lead
            return score

        for a in ACTIONS:
            scores[a] = eval_action(a)
        return scores

    def score_diving(self, gpu, regs):
        scores = {a: 0.0 for a in ACTIONS}
        if gpu == "GAME_OVER":
            self.dive.reset()
            return scores

        self.dive.start_or_sync(gpu)
        if not self.dive.goal:
            return scores

        combo = regs[3 + self.me]
        want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
        want_action = DIRS.get(want, None)

        for a in ACTIONS:
            if a == want_action:
                scores[a] = (combo + 1) * 1.0
            else:
                scores[a] = 0.0
        return scores

    # ---------- Critical gating logic ----------
    def is_hurdles_critical(self, gpu, regs):
        if gpu == "GAME_OVER": return False
        track = gpu
        fin = len(track) - 1
        pos_me = regs[self.me]
        stun = regs[3 + self.me]
        if stun > 0: return False
        dist_me = fin - pos_me
        dist_any = min(fin - regs[0], fin - regs[1], fin - regs[2])
        # Critical if someone <=3 from finish, or we <=4 (one push away), or a hurdle right in front
        hurdle_next = (pos_me + 1 <= fin and track[pos_me + 1] == '#')
        return (dist_any <= 3) or (dist_me <= 4) or hurdle_next

    def is_archery_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        base = 2 * self.me
        x, y = regs[base], regs[base+1]
        dist = abs(x) + abs(y)
        # If only 1-2 winds are visible OR we are still far from the center, push harder
        return (len(gpu) <= 2) or (dist >= 16)

    def is_skating_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or len(gpu) != 4: return False
        risk = regs[3 + self.me]
        turns_left = max(0, regs[6])
        # Near end or near stun boundary is critical
        return (turns_left <= 3) or (risk >= 4)

    def is_diving_critical(self, gpu, regs):
        if gpu == "GAME_OVER" or not gpu: return False
        combo = regs[3 + self.me]
        if not self.dive.goal:
            return False
        steps_left_in_seq = (len(self.dive.goal) - (self.dive.idx % len(self.dive.goal))) % len(self.dive.goal)
        # Critical if big combo (high marginal points) or near end of sequence
        return (combo >= 4) or (steps_left_in_seq <= 2)

    def weight_for_game(self, gi):
        # Base weights with small preference for racing games
        base = [1.1, 0.9, 1.1, 1.0][gi]
        # If points for that game are still zero (no S/G), push harder to avoid zero product
        if gi < len(self.mini_points) and self.mini_points[gi] == 0:
            base += 0.7
        return base

    # ---------- Decision ----------
    def decide(self, games, score_lines):
        self.turn += 1
        self.update_medal_points(score_lines[self.me])

        # Per-game raw scores
        s_h = self.score_hurdles(games[self.G_HURD][0], games[self.G_HURD][1])
        s_a = self.score_archery(games[self.G_ARCH][0], games[self.G_ARCH][1])
        s_s = self.score_skating(games[self.G_SKAT][0], games[self.G_SKAT][1])
        s_d = self.score_diving(games[self.G_DIVE][0], games[self.G_DIVE][1])

        # Normalize
        n_h, n_a, n_s, n_d = map(norm_scores, (s_h, s_a, s_s, s_d))

        # Critical gating
        crit = [
            self.is_hurdles_critical(games[self.G_HURD][0], games[self.G_HURD][1]),
            self.is_archery_critical(games[self.G_ARCH][0], games[self.G_ARCH][1]),
            self.is_skating_critical(games[self.G_SKAT][0], games[self.G_SKAT][1]),
            self.is_diving_critical(games[self.G_DIVE][0], games[self.G_DIVE][1]),
        ]

        # If any game is critical, emphasize only those (winner-take-most)
        action_scores = {a: 0.0 for a in ACTIONS}
        if any(crit):
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                if crit[gi]:
                    w = 2.2 * self.weight_for_game(gi)   # upweight critical ones
                    for a in ACTIONS:
                        action_scores[a] += w * nmap.get(a, 0.0)
        else:
            # Smooth fusion when nothing is urgent
            for gi, nmap in enumerate([n_h, n_a, n_s, n_d]):
                w = self.weight_for_game(gi)
                for a in ACTIONS:
                    action_scores[a] += w * nmap.get(a, 0.0)

        # Final tiebreak: favor the action matching current Diving symbol (keeps combo snowballing)
        dive_bias = 0.0
        want_act = None
        if self.dive.goal:
            want = self.dive.goal[self.dive.idx % len(self.dive.goal)]
            want_act = DIRS.get(want, None)

        best = max(
            ACTIONS,
            key=lambda a: (action_scores[a],
                           0.45 if (want_act and a == want_act) else 0.0,
                           -ACTIONS.index(a))
        )

        # Advance diving index to stay aligned with the run
        if self.dive.running and self.dive.goal:
            self.dive.advance()

        return best

# ---------- I/O loop ----------

def main():
    rd = sys.stdin
    me = int(rd.readline().strip())
    nb = int(rd.readline().strip())

    agent = Agent(me)
    agent.nb_games = nb

    while True:
        # 3 score lines (players 0..2)
        score_lines = []
        for _ in range(3):
            line = rd.readline()
            if not line:
                return
            score_lines.append(line.rstrip("\n"))

        # nb game lines
        games = []
        for _ in range(nb):
            line = rd.readline()
            if not line:
                return
            parts = line.strip().split()
            gpu = parts[0]
            regs = list(map(int, parts[1:8])) if len(parts) >= 8 else [-1]*7
            games.append((gpu, regs))

        act = agent.decide(games, score_lines)
        print(act, flush=True)

if __name__ == "__main__":
    main()

1. Compute per‑game raw scores and normalize to `[0,1]`.
2. Detect **critical games**; use **gating** or **fusion** accordingly.
3. **Tie‑break bias**: add `+0.4` to the action that matches the **current Diving symbol** (helps snowball combos).
4. Final deterministic tie‑break uses a fixed action order to avoid randomness.
5. After printing the action, **advance** the Diving index to remain aligned next turn.

---

## Complexity, Determinism, and Tunables

- **Complexity:** O(1) per turn (constant work, no search).  
- **Determinism:** no randomness; identical states produce identical actions.
- **Tunable constants (behavior knobs):**
  - Hurdles: finish reward `+10`, collision `−9`, progress `+1.2`/tile.
  - Skating: speed `+1.35`/step, collision `−1.0`, adaptive `risk_penalty`/`stun_penalty`.
  - Diving: match reward `combo+1` (scale if desired).
  - Archery: distance heuristic uses Euclidean (can switch to L1).
  - Gating: critical multiplier `2.2 × weight(game)`.
  - Zero‑product guard: extra weight `+0.7` when a game has no S/G medals yet.

---

## Assumptions & Limitations

- **Archery wind parsing:** examples show single‑digit powers; if multi‑digit, the agent uses the **first digit** as current wind.
- **Diving timing:** assumes **one symbol per non‑reset turn**; index advances each turn during a run.
- **Opponent modeling:** only local (e.g., hurdle/position checks, modulo‑10 collision in skating); no deep prediction of opponents’ future actions.
