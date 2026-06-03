# Heuristics

## Hide Agent

Pacman score:

`distance_to_ghost + safe_area + branching_factor + local_area - dead_end_risk - trap_probability`

Features:

- BFS distance to Ghost: prefers real shortest-path separation.
- Safe reachable area: rewards future escape cells.
- Branching factor: avoids low-mobility cells.
- Dead-end distance: penalizes cul-de-sacs when Ghost is close.
- Trap probability: penalizes states where safe flood fill is small.
- Corridor risk: adds pressure penalty for degree-2 corridors.

## Seek Agent

Ghost score:

`-distance_to_pacman + trap_potential + interception_score + cutoff_score - pacman_escape_area`

Features:

- BFS distance to Pacman: direct pursuit.
- Trap potential: rewards Pacman being near low-degree cells.
- Interception score: rewards cells Ghost can reach no later than Pacman.
- Cutoff score: approximates bottleneck control.
- Pacman escape area: penalizes states where Pacman remains mobile.

