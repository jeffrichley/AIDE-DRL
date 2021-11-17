from queue import PriorityQueue
from math import sqrt
import numpy as np

# always have the stay put in position 0
FOUR_CONNECTED_MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

def a_star(start, end, cost):

    if start == end:
        return [end]

    # get the x and y locations
    # adding 1 because we are padding the cost area
    start_y, start_x = start[0] + 1, start[1] + 1
    end_y, end_x = end[0] + 1, end[1] + 1

    # pad the cost matrix so we don't have ugly indexing issues
    cost = np.pad(cost, 1, 'constant', constant_values=999999999)

    open_set = PriorityQueue()
    parents = {}
    visited = []

    # add the start location and a 0 cost
    open_set.put((0, start_y, start_x))
    visited.append((start_y, start_x))
    parents[(start_y, start_x)] = None

    solved = False
    while not solved and not open_set.empty():
        previous_cost, y, x = open_set.get()

        if y == end_y and x == end_x:
            solved = True
            break

        for dir_y, dir_x in FOUR_CONNECTED_MOVES:
            next_y = y + dir_y
            next_x = x + dir_x

            if (next_y, next_x) not in visited:

                g_cost = cost[next_y][next_x]
                h_cost = sqrt((end_y - next_y)**2 + (end_x - next_x)**2)

                open_set.put(((previous_cost + g_cost + h_cost), next_y, next_x))
                parents[(next_y, next_x)] = (y, x)
                visited.append((next_y, next_x))

    path = []
    actions = []
    next = (end_y, end_x)
    while next is not None:
        # don't forget to subtract one because we padded everything
        y = next[0] - 1
        x = next[1] - 1
        path.append((y, x))
        next = parents[next]

    path.reverse()

    here = None
    for next in path:
        if here is None:
            here = next
        else:
            action = (next[0] - here[0], next[1] - here[1])
            action_number = FOUR_CONNECTED_MOVES.index(action)
            actions.append(action_number)
            here = next


    return path, actions

