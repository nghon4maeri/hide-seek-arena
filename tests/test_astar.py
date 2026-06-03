from src.search.bfs import SearchToolkit


def test_astar_path():
    grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    path = SearchToolkit(grid).astar_path((0, 0), (2, 2))
    assert path[-1] == (2, 2)
    assert len(path) == 4

