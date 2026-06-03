# Algorithms

## BFS and UCS

The grid has unit movement cost, so BFS is equivalent to Uniform Cost Search.
`SearchToolkit.bfs_distance_map` computes exact shortest-path distances from one
source to all cells in `O(V + E)` and memoizes the result by source position.

## A*

The Seek Agent uses A* to produce a shortest-path pursuit direction. The
heuristic is Manhattan distance, which is admissible on a four-neighbor grid
without negative costs.

## Flood Fill

Flood fill estimates future mobility. The Hide Agent uses safe flood fill:
a cell counts as safe only if Pacman can arrive before Ghost can catch there.
This approximates escape potential while staying bounded on the 21x21 board.

## Minimax

Both agents use depth-limited minimax. Pacman maximizes survival score. Ghost
maximizes capture score while modeling Pacman's survival response.

## Alpha-Beta

Actions are ordered by heuristic value before search. Good ordering lets
alpha-beta prune branches that cannot change the chosen action.

