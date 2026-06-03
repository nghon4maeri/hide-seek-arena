from src.search.bfs import SearchToolkit


def test_bfs_distance():
    grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    search = SearchToolkit(grid)
    assert search.distance((0, 0), (2, 2)) == 4

