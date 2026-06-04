import type { Grid, Position } from "./types.js";

const MAP_TEXT = `
#####################
#.........#.........#
#.###.###.#.###.###.#
#.###.###.#.###.###.#
#...................#
#.###.#.#####.#.###.#
#.....#...#...#.....#
#####.###.#.###.#####
#####.#...G...#.#####
#####.#.##.##.#.#####
#.......#...#.......#
#####.#.#####.#.#####
#####.#.......#.#####
#####.#.#####.#.#####
#.........#.........#
#.###.###.#.###.###.#
#...#.....P.....#...#
###.#.#.#####.#.#.###
#.....#...#...#.....#
#.#######.#.#######.#
#...................#
#####################
`.trim();

export function parseOfficialMap(): { grid: Grid; pacmanStart: Position; ghostStart: Position } {
  const rows = MAP_TEXT.split("\n");
  let pacmanStart: Position | null = null;
  let ghostStart: Position | null = null;
  const grid = rows.map((row, r) =>
    [...row].map((char, c) => {
      if (char === "#") return 1;
      if (char === "P") pacmanStart = [r, c];
      if (char === "G") ghostStart = [r, c];
      return 0;
    })
  ) as Grid;
  if (!pacmanStart || !ghostStart) throw new Error("Official map is missing P or G");
  return { grid, pacmanStart, ghostStart };
}
