# Submission Notes

## Hide Strategy

The Hide Agent maximizes survival. It prefers cells with high BFS distance from
Ghost, large safe flood-fill area, high branching factor, and strong future
mobility. It penalizes low-degree corridors, nearby dead ends, and states where
Ghost can rapidly collapse the escape region.

Early game behavior emphasizes open-area control. Mid game behavior preserves
multiple escape routes. Late game behavior is conservative and avoids entering
dead ends unless the heuristic shows they remain safe.

## Seek Strategy

The Seek Agent combines A* pursuit with trap pressure. It follows shortest paths
when Pacman is exposed, but it can prefer cutoff moves when a direct chase leaves
Pacman too much safe area. It rewards positions that control corridor mouths,
reduce Pacman's future mobility, and create interception opportunities.

## BFS, A*, and Flood Fill Usage

BFS is used as unit-cost UCS to compute exact grid distances. These distance maps
are cached by source position because the map is static.

A* uses Manhattan distance to guide shortest-path pursuit. It gives Ghost a fast
baseline chase direction and provides path data for debugging.

Flood fill estimates reachable area. The Hide Agent uses safe flood fill, where a
cell counts only if Pacman can reach it before Ghost can catch there.

## Minimax and Alpha-Beta Usage

Both agents use depth-limited minimax. Pacman maximizes survival score while
modeling Ghost's best chase response. Ghost maximizes capture pressure while
modeling Pacman's best escape response.

Alpha-beta pruning cuts branches that cannot change the selected action. Action
ordering expands promising moves first, improving pruning under the one-second
limit.

## Heuristic Evaluation

Hide score combines:

- BFS distance to Ghost.
- Safe reachable area.
- Branching factor.
- Local reachable area.
- Dead-end and corridor penalties.
- Trap probability penalty.

Seek score combines:

- Negative BFS distance to Pacman.
- Trap potential.
- Interception score.
- Cutoff score.
- Penalty for Pacman's safe escape area.

## Complexity and Runtime Safety

The arena has at most 441 cells. BFS and flood fill are `O(V + E)`, A* is
`O(E log V)`, and cached map analysis avoids repeated full recomputation.

Minimax is bounded to depth 3-4 with branch caps, action ordering, alpha-beta
pruning, and timer checks. Normal tournament calls do not create debug traces or
load UI modules. This keeps the agent deterministic, CPU-only, and safe for the
one-second action limit.
