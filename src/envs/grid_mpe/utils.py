from munkres import Munkres, print_matrix


def matched(matrix, n_agents):
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
    return total == n_agents
    