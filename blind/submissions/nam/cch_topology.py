"""topology.py — Static topology analysis of the arena map.

Classification produced once at init:
  - degree / junctions (>=3 exits) / dead ends (<=1)
  - corridor cells (chains between dead ends and junctions)
  - core (iterative leaf trimming — the loop-rich safe region)
  - loops (large cycles inside the core)
  - distance maps to the nearest junction / loop
"""

from collections import deque


class Topology:
    def __init__(self, grid):
        self.grid = grid
        self.junctions = set()
        self.dead_ends = set()
        self.corridors = set()
        self.core = set()
        self.loops = set()
        self.junction_dist = {}
        self.loop_dist = {}
        self.dead_end_depth = {}
        self._analyze()

    def _analyze(self):
        g = self.grid
        free = set(g.free)
        degree = {}
        for r, c in free:
            d = 0
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if g.walkable((r + dr, c + dc)):
                    d += 1
            degree[(r, c)] = d
            if d >= 3:
                self.junctions.add((r, c))
            elif d <= 1:
                self.dead_ends.add((r, c))

        # Corridors: trace from every dead end until a junction.
        for seed in self.dead_ends:
            cur, prev = seed, None
            for _ in range(25):
                if cur not in self.junctions:
                    self.corridors.add(cur)
                else:
                    break
                nxts = [n for n in g.neighbors(cur) if n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in self.corridors and nxt not in self.junctions:
                    break
                prev, cur = cur, nxt

        # Dead-end depth.
        for de in self.dead_ends:
            cur, depth = de, 0
            visited = {de}
            while cur not in self.junctions:
                moved = False
                for nxt in g.neighbors(cur):
                    if nxt not in visited:
                        visited.add(nxt)
                        cur, depth, moved = nxt, depth + 1, True
                        break
                if not moved or depth > 25:
                    break
            self.dead_end_depth[de] = min(depth, 10)

        # Core: iterative leaf trimming.
        active = set(free)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                na = sum(1 for n in g.neighbors(cell) if n in active)
                if na <= 1:
                    to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self.core = active

        # Loops: DFS back-edge cycles inside the core, keep the biggest few.
        self.loops = self._find_cycles()

        # Distance maps.
        self.junction_dist = self._bfs_map(self.junctions if self.junctions else self.core)
        self.loop_dist = self._bfs_map(self.loops if self.loops else self.core)

    def _find_cycles(self):
        g = self.grid
        core = self.core
        cycles = []
        visited = set()
        parent = {}
        for start in core:
            if start in visited:
                continue
            stack = [(start, None)]
            visited.add(start)
            parent[start] = None
            while stack:
                u, p = stack.pop()
                for v in g.neighbors(u):
                    if v not in core:
                        continue
                    if v not in visited:
                        visited.add(v)
                        parent[v] = u
                        stack.append((v, u))
                    elif v != p:
                        cycle = [v]
                        cur = u
                        while cur is not None and cur != v:
                            cycle.append(cur)
                            cur = parent.get(cur)
                        if cur == v and len(cycle) >= 6:
                            cycles.append(set(cycle))
        cycles.sort(key=len, reverse=True)
        out = set()
        for cyc in cycles[:4]:
            out |= cyc
        return out

    def _bfs_map(self, starts):
        g = self.grid
        dist = {p: 10 ** 9 for p in g.free}
        q = deque()
        for p in starts:
            dist[p] = 0
            q.append(p)
        while q:
            cur = q.popleft()
            for nxt in g.neighbors(cur):
                if dist[nxt] > dist[cur] + 1:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def weight_of(self, cell):
        """Topological attractiveness of a cell (higher = safer to be)."""
        if cell in self.junctions:
            return 3.0
        if cell in self.loops:
            return 2.5
        if cell in self.core:
            return 2.0
        if cell in self.corridors:
            return 0.4
        if cell in self.dead_ends:
            return 0.05
        return 1.0
