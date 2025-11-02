# Ultimate Tic-Tac-Toe Bot (CodinGame)

A strong **Ultimate Tic-Tac-Toe** bot for CodinGame.  
It combines **fast tactical rules** (instant wins/blocks, trap avoidance) with a **PUCT-style Monte Carlo Tree Search** that runs under strict time limits.

## TL;DR (How it wins)
1. **Instant win** if a move wins a small board *and* completes 3 in a row on the macro board.  
2. Otherwise **take any instant mini-board win**, preferring ones that strengthen macro lines.  
3. **Avoid traps**: don’t send the opponent to a small board where they have an immediate win.  
4. If none of the above decides it, run **MCTS (PUCT)** for ~85 ms to explore the best futures and pick the most visited move.

## Game Rules (quick recap)
- The game is a 3×3 grid of small tic-tac-toe boards (9 total).
- Playing in a small board cell **forces your opponent** to play next in the **corresponding** small board on the macro grid.
- If that target small board is **full or already won**, the opponent can play **anywhere**.
- You win the big game by winning **3 small boards in a row** (macro rows/cols/diagonals).
- If no macro 3-in-a-row is achieved and the big board ends, **most small-board wins** decides.

## I/O (CodinGame “agent” format)
**Input per turn**
```
opponentRow opponentCol    # -1 -1 on your first turn
validActionCount
row col
row col
...
```

**Output per turn**
```
row col
```

- Rows/cols are **global** coordinates in `[0..8]`.
- CodinGame already filters **valid moves** for the current turn (respecting the forced small board rule).

## How to Run on CodinGame
1. Create a new bot with the Python template.
2. Paste the full Python file (this bot) into `main.py` (or similar).
3. Click **Test** / **Submit**.

> The bot respects time constraints: ≈900 ms on the first move, ≈85 ms on later moves.

## How it’s implemented (short version)
- **Board state**
  - `board`: list of 81 integers (`0` empty, `+1` us, `-1` opponent`).
  - `sb_status`: list of 9 integers (`0` ongoing, `+1` we won, `-1` they won, `2` draw`).
- **Tactical layer**
  - Checks for immediate **macro wins** via a mini win.
  - Picks **instant mini wins** and rates them by macro impact.
  - Applies a **danger penalty** if a move sends the opponent to a mini-board they can win immediately.
- **MCTS layer (PUCT)**
  - **Priors** prefer center/corners, instant mini wins, and macro-helpful boards.
  - **Selection** uses PUCT: `score = Q + cPUCT * P * sqrt(ln(N_parent)/(1+N_child))`.
  - **Expansion** adds a child on the first unexpanded move encountered.
  - **Rollout** uses a **fast heuristic** (not random) to finish quickly.
  - **Backpropagation** accumulates results from **our** point of view.
  - Root move is the **most visited** child (“robust child”).

## Tuning Knobs
You can tweak a few constants in the code:
- **Time budgets**: `~900 ms` first move, `~85 ms` others.
- **cPUCT**: exploration vs exploitation (≈ `1.1–1.5` is a good range).
- **Danger penalty**: how strongly to avoid sending the opponent to instant-win boards.
- **Rollout depth**: small increases can help; keep an eye on timing.

## Local Testing (optional)
You can stub a simple driver that feeds the bot lines like:
```
-1 -1
5
3 3
3 4
4 3
4 4
4 5
```
and capture its output `(row col)`.  
But the easiest is to use CodinGame’s IDE and test harness.

## Troubleshooting
- **Timeouts**: reduce budgets or rollout length slightly.
- **Still losing to a specific style**:
  - Increase **danger penalty** if you get forked a lot.
  - Increase **prior** bonus for instant mini wins (search focuses faster).
  - Slightly increase **cPUCT** to explore more.
- **Weird moves on free-choice turns**: check that `forced` is `None` when the target board is full/won.

## Why MCTS here?
Ultimate Tic-Tac-Toe has **branching traps**. Shallow minimax misses a lot.  
MCTS with good priors and safe rollouts **allocates time** to the most promising lines and handles tactics + strategy at once.

## License
MIT
