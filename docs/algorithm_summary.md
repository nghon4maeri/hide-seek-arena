# Algorithm Summary

This document summarizes algorithms relevant to the Hide and Seek Arena lab. It does not claim every listed algorithm is fully implemented in this `pacman/` workspace.

## BFS

Breadth-First Search explores states by increasing depth using a queue. On a unit-cost grid, BFS finds shortest paths by number of moves.

Typical use in this lab:

- Shortest-path distance.
- Reachability checks.
- Local safety and mobility analysis.

## A*

A* Search uses both path cost and a heuristic estimate:

```text
f(n) = g(n) + h(n)
```

For grid movement, Manhattan distance is a common heuristic. A* is useful for pursuing a visible target or planning a path through the maze.

## Flood Fill

Flood fill expands through connected passable cells to estimate reachable area. In a Hide strategy, it can help avoid dead ends and positions with low future mobility.

## Minimax

Minimax models the game as an adversarial decision problem. One player chooses a move assuming the opponent will respond with a move that is best for the opponent.

In this lab, minimax is usually depth-limited because a full game tree is too large for per-step decisions.

## Alpha-Beta Pruning

Alpha-beta pruning improves minimax by skipping branches that cannot affect the final decision. It does not change the minimax result at the same search depth; it only reduces unnecessary evaluation.

## Monte Carlo

Monte Carlo methods estimate outcomes by sampling random or semi-random simulations. They can be useful when exact search is too expensive.

Current status: this workspace setup does not add Monte Carlo logic. Do not claim Monte Carlo is implemented unless a future agent actually adds it.

## Expectiminimax

Expectiminimax extends minimax for games with uncertainty or probabilistic events. It uses expected values for chance nodes.

Current status: this workspace setup does not add Expectiminimax logic. It may be considered later only if the game configuration introduces uncertainty that should be modeled explicitly.

## Agent Usage Summary

| Concept | Hide Agent Use | Seek Agent Use |
| --- | --- | --- |
| BFS | Escape distance, safe area, dead-end checks | Distance to target and interception checks |
| A* | Optional escape route planning | Target pursuit path planning |
| Flood Fill | Survival and mobility estimate | Trap pressure estimate |
| Minimax | Model future Seek responses | Model future Hide responses |
| Alpha-Beta | Reduce minimax branch cost | Reduce minimax branch cost |
| Monte Carlo | Not added by this setup | Not added by this setup |
| Expectiminimax | Not added by this setup | Not added by this setup |
