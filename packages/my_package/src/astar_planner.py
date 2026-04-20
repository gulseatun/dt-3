import math

COORDS = {
    0 : (0,0),
    1 : (1,0),
    2 : (2,0),
    3 : (3,0),
    4 : (0,1),
    5 : (1,1),
    6 : (2,1),
    7 : (3,1),
    8 : (0,2),
    9 : (1,2),
    10 : (2,2),
    11 : (3,2),
    12 : (0,3),
    13 : (1,3),
    14 : (2,3),
    15 : (3,3)
}

GRAPH = {
    0 : {1: 1.5, 4: 2.0},
    1 : {0: 1.5, 2: 1.0, 5: 2.0},
    2 : {1: 1.0, 3: 1.0, 6: 1.5},
    3 : {2: 1.0},
    4 : {0: 2.0, 8: 1.5},
    5 : {1: 2.0, 6: 1.0, 9: 2.0},
    6 : {2: 1.5, 5: 1.0, 7: 0.5, 10: 4.0},
    7 : {6: 0.5, 11: 1.5},
    8 : {4: 1.5, 9: 1.5, 12: 2.0},
    9 : {5: 2.0, 8: 1.5, 10: 2.0},
    10 : {6: 4.0, 9: 2.0, 11: 1.0, 14: 1.5},
    11 : {7: 1.5, 10: 1.0},
    12 : {8: 2.0, 13: 1.5},
    13 : {12: 1.5, 14: 2.0},
    14 : {10: 1.5, 13: 2.0, 15: 1.0},
    15 : {14: 1.0}
}

def heuristic(node, goal, method="manhattan"):

    x1, y1 = COORDS[node]
    x2, y2 = COORDS[goal]
    
    if method == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)
    
    elif method == "euclidean":
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def a_star_search(start, goal):

    if start not in COORDS or goal not in COORDS:
        return [], float('inf')

    if start == goal:
        return [start], 0.0

    OPEN = [(heuristic(start, goal), heuristic(start, goal), start)]
    CLOSED = set()

    g_costs = {node: float('inf') for node in COORDS.keys()}
    g_costs[start] = 0
    parent = {start: None}

    while OPEN:
        
        OPEN.sort(key=lambda x: (x[0], x[1]))

        _, _, current_node = OPEN.pop(0)

        if current_node == goal:

            print("Hedef Bulundu! Yol hesaplanıyor...\n")
            return reconstruct_path(parent, current_node), g_costs[goal]
        
        CLOSED.add(current_node)

        for neighbor, move_cost in GRAPH[current_node].items():

            if neighbor in CLOSED:
                continue
            
            temporary_g_cost = g_costs[current_node] + move_cost

            if temporary_g_cost < g_costs.get(neighbor, float('inf')):

                parent[neighbor] = current_node
                g_costs[neighbor] = temporary_g_cost
                h_val = heuristic(neighbor, goal)
                f_val = temporary_g_cost + h_val

                OPEN = [item for item in OPEN if item[2] != neighbor]
                OPEN.append((f_val, h_val, neighbor))

    return [], float('inf')

def reconstruct_path(parent, current):

    total_path = [current]

    while current in parent and parent[current] is not None:
        current = parent[current]
        total_path.insert(0, current)

    return total_path

if __name__ == "__main__":
    START_NODE = 0
    GOAL_NODE = 15
    
    print("A* Algoritması çalışıyor...\n")
    path, total_cost = a_star_search(START_NODE, GOAL_NODE)
    
    if path:
        formatted_path = " -> ".join([f"N{node}" for node in path])
        
        print("--- SONUÇLAR ---\n")
        print(f"Hesaplanan Rota : {formatted_path}")
        print(f"Toplam Maliyet  : {total_cost:.1f}")
    else:
        print("Hedefe giden hiçbir yol bulunamadı!")