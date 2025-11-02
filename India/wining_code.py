# Ultimate Tic-Tac-Toe bot (PUCT-MCTS + tactical rules) for CodinGame I/O
# Turn input:
#   Line 1: opponentRow opponentCol  (-1 -1 on the very first turn)
#   Line 2: validActionCount
#   Next validActionCount lines: row col   (global coords 0..8)
# Output:
#   row col
#
# Strategy:
#   - Pre-move tactics: take instant macro win; else instant mini win (macro-aware);
#     avoid sending opponent to an instant-win subboard.
#   - PUCT-MCTS with policy priors & deadline (≈900ms first turn, ≈85ms others).
#   - Heuristic rollouts if we don't reach a terminal state quickly.

import sys, time, math, random
from typing import List, Tuple, Optional, Dict

ME, OP, DRAW = 1, -1, 2

# ---------- Precomputed geometry ----------
ALL_RC = [(r, c) for r in range(9) for c in range(9)]
IDX = {(r, c): r * 9 + c for r in range(9) for c in range(9)}
RC = [(i // 9, i % 9) for i in range(81)]

def mcoords(i: int) -> Tuple[int, int]:
    r, c = RC[i]
    return r // 3, c // 3

def lcoords(i: int) -> Tuple[int, int]:
    r, c = RC[i]
    return r % 3, c % 3

# 3x3 winning lines (triplets of (r,c) local coords)
LINES3 = [
    [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],  # rows
    [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],  # cols
    [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]                        # diags
]

# For each subboard (0..8), the 9 cell indices
SUB_CELLS: List[List[int]] = []
SUB_LINES: List[List[Tuple[int,int,int]]] = []  # as tuples of global idx
for br in range(3):
    for bc in range(3):
        cells = []
        for dr in range(3):
            for dc in range(3):
                r = br*3 + dr
                c = bc*3 + dc
                cells.append(IDX[(r,c)])
        SUB_CELLS.append(cells)
        # lines as global index triplets
        glines = []
        for line in LINES3:
            a = IDX[(br*3+line[0][0], bc*3+line[0][1])]
            b = IDX[(br*3+line[1][0], bc*3+line[1][1])]
            c = IDX[(br*3+line[2][0], bc*3+line[2][1])]
            glines.append((a,b,c))
        SUB_LINES.append(glines)

# Macro winning lines as subboard indices 0..8
MACRO_LINES: List[Tuple[int,int,int]] = []
for line in LINES3:
    ids = []
    for (r,c) in line:
        ids.append(r*3 + c)
    MACRO_LINES.append(tuple(ids))

# Preferred local positions (center > corners > edges)
PREF_LOCAL = [(1,1), (0,0),(0,2),(2,0),(2,2), (0,1),(1,0),(1,2),(2,1)]


# ---------- Game state helpers ----------
def micro_winner(board: List[int], sb_status: List[int], sb: int) -> int:
    """Return ME/OP/DRAW/0 (ongoing) for subboard sb."""
    if sb_status[sb] != 0:
        return sb_status[sb]
    cells = SUB_CELLS[sb]
    lines = SUB_LINES[sb]
    for (a,b,c) in lines:
        s = board[a] + board[b] + board[c]
        if s == 3:  return ME
        if s == -3: return OP
    # Check full
    for i in cells:
        if board[i] == 0:
            return 0
    return DRAW

def update_sb_after_move(board: List[int], sb_status: List[int], move_idx: int):
    sb = (move_idx // 9) // 3 * 3 + ((move_idx % 9) // 3)  # but simpler:
    br, bc = mcoords(move_idx)
    sb = br*3 + bc
    sb_status[sb] = micro_winner(board, sb_status, sb)

def macro_winner(sb_status: List[int]) -> int:
    for (a,b,c) in MACRO_LINES:
        va, vb, vc = sb_status[a], sb_status[b], sb_status[c]
        if va == vb == vc == ME: return ME
        if va == vb == vc == OP: return OP
    return 0

def sub_full(board: List[int], sb: int) -> bool:
    for i in SUB_CELLS[sb]:
        if board[i] == 0:
            return False
    return True

def board_full(board: List[int]) -> bool:
    return all(v != 0 for v in board)

def legal_moves(board: List[int], sb_status: List[int], forced: Optional[int]) -> List[int]:
    """Return legal moves (global idx)."""
    if forced is not None and sb_status[forced] == 0 and not sub_full(board, forced):
        return [i for i in SUB_CELLS[forced] if board[i] == 0]
    # free choice over ongoing subboards
    res = []
    for sb in range(9):
        if sb_status[sb] == 0:
            for i in SUB_CELLS[sb]:
                if board[i] == 0:
                    res.append(i)
    return res

def next_forced(move_idx: int, sb_status: List[int], board: List[int]) -> Optional[int]:
    lr, lc = lcoords(move_idx)
    forced = lr*3 + lc
    if sb_status[forced] != 0 or sub_full(board, forced):
        return None
    return forced

def immediate_mini_wins(board: List[int], sb_status: List[int], player: int, in_sb: int) -> List[int]:
    """All moves inside subboard in_sb that instantly win it for 'player'."""
    if in_sb is None or sb_status[in_sb] != 0:
        return []
    res = []
    for (a,b,c) in SUB_LINES[in_sb]:
        line = (a,b,c)
        s = board[a] + board[b] + board[c]
        if (player == ME and s == 2) or (player == OP and s == -2):
            for i in line:
                if board[i] == 0:
                    res.append(i)
    return list(set(res))

def danger_penalty_after_move(board: List[int], sb_status: List[int], move_idx: int) -> int:
    """If our move sends opponent to a subboard they can instantly win, penalize."""
    forced = next_forced(move_idx, sb_status, board)
    if forced is None:
        return 0
    wins = immediate_mini_wins(board, sb_status, OP, forced)
    return -1200 * len(wins)

def terminal_score(board: List[int], sb_status: List[int]) -> Optional[int]:
    """Return final result from ME's perspective if terminal, else None."""
    mw = macro_winner(sb_status)
    if mw == ME: return 1_000_000
    if mw == OP: return -1_000_000
    if board_full(board):
        # tie-break by # won subboards
        me_w = sum(1 for s in sb_status if s == ME)
        op_w = sum(1 for s in sb_status if s == OP)
        if me_w > op_w: return 100_000 + (me_w - op_w)
        if op_w > me_w: return -100_000 - (op_w - me_w)
        return 0
    return None

def heuristic(board: List[int], sb_status: List[int]) -> int:
    """Non-terminal eval: macro > micro."""
    mw = macro_winner(sb_status)
    if mw == ME: return 900_000
    if mw == OP: return -900_000

    score = 0

    # Macro lines
    for (a,b,c) in MACRO_LINES:
        vals = [sb_status[a], sb_status[b], sb_status[c]]
        if DRAW in vals:  # dead line
            continue
        me_cnt = vals.count(ME)
        op_cnt = vals.count(OP)
        empty  = 3 - me_cnt - op_cnt
        if me_cnt > 0 and op_cnt == 0:
            if me_cnt == 2 and empty == 1: score += 6000
            elif me_cnt == 1 and empty == 2: score += 250
        elif op_cnt > 0 and me_cnt == 0:
            if op_cnt == 2 and empty == 1: score -= 7000
            elif op_cnt == 1 and empty == 2: score -= 300

    # Micro features
    for sb in range(9):
        if sb_status[sb] != 0:  # decided
            continue
        # 2-in-row threats
        for (a,b,c) in SUB_LINES[sb]:
            vals = [board[a], board[b], board[c]]
            me_cnt = vals.count(ME)
            op_cnt = vals.count(OP)
            empty  = vals.count(0)
            if me_cnt > 0 and op_cnt == 0:
                if me_cnt == 2 and empty == 1: score += 45
                elif me_cnt == 1 and empty == 2: score += 10
            elif op_cnt > 0 and me_cnt == 0:
                if op_cnt == 2 and empty == 1: score -= 55
                elif op_cnt == 1 and empty == 2: score -= 12
        # center control (mild)
        center_idx = SUB_CELLS[sb][4]
        if board[center_idx] == ME: score += 6
        elif board[center_idx] == OP: score -= 6

    return score

# ---------- PUCT-MCTS ----------
class Node:
    __slots__ = ("player", "forced", "N", "W", "P", "children", "moves")

    def __init__(self, player: int, forced: Optional[int], moves: List[int], priors: Dict[int, float]):
        self.player = player
        self.forced = forced
        self.N = 0
        self.W = 0.0  # sum of rewards from ME's perspective
        self.P = {m: priors.get(m, 1e-6) for m in moves}  # prior policy
        self.children: Dict[int, Node] = {}
        self.moves = moves

    def best_child_for_play(self):
        # choose by visit count (robust child)
        return max(self.children.items(), key=lambda kv: kv[1].N)[0]

def policy_prior(board: List[int], sb_status: List[int], moves: List[int]) -> Dict[int, float]:
    """Light heuristic prior: immediate mini wins >> good local shapes."""
    pri = {}
    for m in moves:
        br, bc = mcoords(m)
        sb = br*3 + bc
        lr, lc = lcoords(m)
        loc = (lr, lc)
        p = 1.0
        # strong for instant mini-wins
        if sb_status[sb] == 0:
            # test win
            board[m] = ME
            prev = sb_status[sb]
            sb_status[sb] = micro_winner(board, sb_status, sb)
            if sb_status[sb] == ME:
                p += 12.0
            sb_status[sb] = prev
            board[m] = 0
        # local preference
        if loc == (1,1): p += 2.0
        elif loc in [(0,0),(0,2),(2,0),(2,2)]: p += 1.2
        else: p += 0.6
        # macro impact (lines where this subboard sits)
        if sb_status[sb] == 0:
            add = 0.0
            for (a,b,c) in MACRO_LINES:
                if sb in (a,b,c):
                    vals = [sb_status[a], sb_status[b], sb_status[c]]
                    if DRAW in vals: 
                        continue
                    me_cnt = vals.count(ME)
                    op_cnt = vals.count(OP)
                    empty  = 3 - me_cnt - op_cnt
                    if me_cnt == 1 and empty == 2:
                        add += 0.5
                    elif me_cnt == 2 and empty == 1:
                        add += 1.5
        else:
            add = 0.1
        pri[m] = p + add
    # Normalize
    s = sum(pri.values())
    if s > 0:
        for k in pri:
            pri[k] /= s
    return pri

def puct_select(node: Node, parent_ln: float) -> int:
    """Select move by PUCT: Q + c * P * sqrt(ln(N_parent+1)/(N_child+1))."""
    c_puct = 1.25
    best, best_score = None, -1e18
    for m in node.moves:
        child = node.children.get(m)
        if child is None:
            q = 0.0
            n = 0
        else:
            q = child.W / (child.N + 1e-9)
            n = child.N
        u = c_puct * node.P[m] * math.sqrt(parent_ln / (n + 1))
        val = q + u
        if val > best_score:
            best_score, best = val, m
    return best

def simulate_rollout(board: List[int], sb_status: List[int], player: int, forced: Optional[int], limit: int = 28) -> int:
    """Heuristic playout from current state; return ME-perspective score."""
    # light tactical policy during rollout
    for _ in range(limit):
        term = terminal_score(board, sb_status)
        if term is not None:
            return term
        moves = legal_moves(board, sb_status, forced)
        if not moves:
            return heuristic(board, sb_status)

        # 1) play instant mini-win if exists
        if forced is not None and sb_status[forced] == 0:
            wins = immediate_mini_wins(board, sb_status, player, forced)
            if wins:
                m = random.choice(wins)
            else:
                # Avoid sending opponent to instant win if possible
                cands = []
                worst = []
                for m in moves:
                    board[m] = player
                    br, bc = mcoords(m)
                    sb = br*3 + bc
                    prev = sb_status[sb]
                    update_sb_after_move(board, sb_status, m)
                    pen = danger_penalty_after_move(board, sb_status, m)
                    nxt = next_forced(m, sb_status, board)
                    board[m] = 0
                    sb_status[sb] = prev
                    (cands if pen == 0 else worst).append(m)
                pool = cands if cands else moves
                # local pref
                pool.sort(key=lambda x: PREF_LOCAL.index(lcoords(x)) if lcoords(x) in PREF_LOCAL else 9)
                m = pool[0]
        else:
            # free choice: prefer instant mini-wins anywhere
            anywins = []
            for sb in range(9):
                if sb_status[sb] != 0: continue
                anywins.extend(immediate_mini_wins(board, sb_status, player, sb))
            if anywins:
                m = random.choice(anywins)
            else:
                # mild heuristic: prefer center/corners; avoid obvious danger
                safe = []
                for m_ in moves:
                    board[m_] = player
                    br, bc = mcoords(m_)
                    sb = br*3 + bc
                    prev = sb_status[sb]
                    update_sb_after_move(board, sb_status, m_)
                    pen = danger_penalty_after_move(board, sb_status, m_)
                    board[m_] = 0
                    sb_status[sb] = prev
                    if pen == 0:
                        safe.append(m_)
                pool = safe if safe else moves
                pool.sort(key=lambda x: PREF_LOCAL.index(lcoords(x)) if lcoords(x) in PREF_LOCAL else 9)
                m = pool[0]

        # apply
        board[m] = player
        br, bc = mcoords(m)
        sb = br*3 + bc
        prev = sb_status[sb]
        update_sb_after_move(board, sb_status, m)
        forced = next_forced(m, sb_status, board)
        player = -player

    # depth limit hit -> evaluate
    return heuristic(board, sb_status)

def mcts_root(board: List[int], sb_status: List[int], root_moves: List[int], deadline_s: float) -> int:
    # Root forced?
    minis = {mcoords(m)[0]*3 + mcoords(m)[1] for m in root_moves}
    root_forced = list(minis)[0] if len(minis) == 1 else None

    pri = policy_prior(board, sb_status, root_moves)
    root = Node(ME, root_forced, root_moves, pri)

    # Run simulations until deadline
    while time.perf_counter() < deadline_s:
        # Copy state
        b = board[:]             # 81 ints
        s = sb_status[:]         # 9 ints
        path: List[Tuple[Node,int,int,int]] = []  # (node, move, brbc, prev_status)

        node = root
        player = ME
        forced = root_forced

        # --- Selection & Expansion ---
        while True:
            term = terminal_score(b, s)
            if term is not None:
                reward = term
                break

            if not node.moves:  # no legal moves (shouldn't happen)
                reward = heuristic(b, s)
                break

            # If an unexpanded move exists, expand it
            expandable = [m for m in node.moves if m not in node.children]
            if expandable:
                m = random.choice(expandable)
                # apply move
                b[m] = node.player
                br, bc = mcoords(m)
                sb = br*3 + bc
                prev = s[sb]
                update_sb_after_move(b, s, m)
                nf = next_forced(m, s, b)

                # child legal moves for next player
                child_moves = legal_moves(b, s, nf)
                # child priors
                child_pri = policy_prior(b, s, child_moves)
                child = Node(-node.player, nf, child_moves, child_pri)
                node.children[m] = child
                path.append((node, m, sb, prev))
                node = child
                player = node.player
                forced = nf
                # rollout from here
                reward = simulate_rollout(b, s, player, forced, limit=24)
                break
            else:
                # select by PUCT
                parent_ln = math.log(node.N + 1.0)
                m = puct_select(node, parent_ln if parent_ln > 0 else 1.0)
                # apply
                b[m] = node.player
                br, bc = mcoords(m)
                sb = br*3 + bc
                prev = s[sb]
                update_sb_after_move(b, s, m)
                nf = next_forced(m, s, b)
                path.append((node, m, sb, prev))
                node = node.children[m]
                player = node.player
                forced = nf

        # --- Backpropagate (ME perspective) ---
        val = reward
        for (n, m, sb, prev_stat) in reversed(path):
            n.N += 1
            # If node.player is the actor at that node, from ME's perspective:
            # reward already in ME frame; nothing to flip here.
            n.W += val
            # Undo in our copy (not strictly required post-backprop in this implementation)

    # Pick move with highest visits; if tie, best average value
    if root.children:
        # Robust child
        best_m = max(root.children.items(), key=lambda kv: (kv[1].N, kv[1].W/(kv[1].N+1e-9)))[0]
        return best_m
    # Fallback
    return root_moves[0]

# ---------- Move picker (tactics + MCTS) ----------
def pick_move(board: List[int], sb_status: List[int], valid_moves_rc: List[Tuple[int,int]], deadline_s: float) -> Tuple[int,int]:
    valid_moves = [IDX[(r,c)] for (r,c) in valid_moves_rc]

    # 0) Instant macro win via mini win
    for m in valid_moves:
        br, bc = mcoords(m)
        sb = br*3 + bc
        if sb_status[sb] != 0:
            continue
        board[m] = ME
        prev = sb_status[sb]
        update_sb_after_move(board, sb_status, m)
        if sb_status[sb] == ME and macro_winner(sb_status) == ME:
            board[m] = 0
            sb_status[sb] = prev
            return RC[m]
        board[m] = 0
        sb_status[sb] = prev

    # 1) Immediate mini win (macro-aware)
    best_w, best_mv = -10**9, None
    for m in valid_moves:
        br, bc = mcoords(m)
        sb = br*3 + bc
        if sb_status[sb] != 0:  # already decided
            continue
        board[m] = ME
        prev = sb_status[sb]
        update_sb_after_move(board, sb_status, m)
        if sb_status[sb] == ME:
            bonus = 0
            for (a,b,c) in MACRO_LINES:
                if sb in (a,b,c):
                    vals = [sb_status[a], sb_status[b], sb_status[c]]
                    if DRAW in vals: continue
                    me_cnt = vals.count(ME)
                    op_cnt = vals.count(OP)
                    empty  = 3 - me_cnt - op_cnt
                    if me_cnt == 2 and empty == 1:
                        bonus += 1500
                    elif me_cnt == 1 and empty == 2:
                        bonus += 200
            # prefer safer sends
            bonus += danger_penalty_after_move(board, sb_status, m)
            if bonus > best_w:
                best_w, best_mv = bonus, m
        board[m] = 0
        sb_status[sb] = prev
    if best_mv is not None:
        return RC[best_mv]

    # 2) Filter obviously losing sends if we have alternatives
    safe = []
    for m in valid_moves:
        board[m] = ME
        br, bc = mcoords(m)
        sb = br*3 + bc
        prev = sb_status[sb]
        update_sb_after_move(board, sb_status, m)
        pen = danger_penalty_after_move(board, sb_status, m)
        board[m] = 0
        sb_status[sb] = prev
        if pen == 0:
            safe.append(m)
    root_moves = safe if safe else valid_moves

    # 3) PUCT-MCTS among root_moves
    move_idx = mcts_root(board, sb_status, root_moves, deadline_s)
    return RC[move_idx]

# ---------- Main loop ----------
def main():
    board = [0]*81            # 0 empty, +1 ME, -1 OP
    sb_status = [0]*9         # per subboard: 0 ongoing, +1 ME, -1 OP, 2 DRAW
    turn_idx = 0

    while True:
        try:
            opp_r, opp_c = map(int, input().split())
        except EOFError:
            break

        if opp_r != -1:
            opi = IDX[(opp_r, opp_c)]
            board[opi] = OP
            update_sb_after_move(board, sb_status, opi)

        try:
            n = int(input())
        except:
            n = 0
        valid_moves_rc: List[Tuple[int,int]] = []
        for _ in range(n):
            r, c = map(int, input().split())
            valid_moves_rc.append((r, c))

        if not valid_moves_rc:
            print("0 0")
            sys.stdout.flush()
            continue

        now = time.perf_counter()
        budget_ms = 900 if turn_idx == 0 else 85
        deadline_s = now + budget_ms / 1000.0

        r, c = pick_move(board, sb_status, valid_moves_rc, deadline_s)

        # Apply our move locally
        mi = IDX[(r, c)]
        board[mi] = ME
        update_sb_after_move(board, sb_status, mi)

        print(r, c)
        sys.stdout.flush()
        turn_idx += 1

if __name__ == "__main__":
    main()
