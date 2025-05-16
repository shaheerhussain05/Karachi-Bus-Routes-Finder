# Fixed and improved parts of your route-finding code

import json
import math
import random
import heapq
from collections import deque
from typing import List, Optional, Tuple

# --------------------------------
# Load and Normalize Data
# --------------------------------

def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())  # removes extra spaces and trims

with open("cleaned_karachi_graph.json") as f:
    raw_graph = json.load(f)

with open("location_coords.json") as f:
    raw_coords = json.load(f)

# Normalize coordinates
coords = {normalize_name(k): v for k, v in raw_coords.items() if v}

# Normalize graph
karachi_graph = {}
for raw_node, data in raw_graph.items():
    norm_node = normalize_name(raw_node)
    norm_connected = [normalize_name(n) for n in data.get("connected", [])]
    norm_buses = list(set(data.get("buses", [])))
    karachi_graph[norm_node] = {
        "connected": norm_connected,
        "buses": norm_buses
    }

# --------------------------------
# Helper Functions
# --------------------------------

def euclidean_distance(coord1: List[float], coord2: List[float]) -> float:
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def heuristic(current: str, goal: str) -> float:
    if current in coords and goal in coords:
        return euclidean_distance(coords[current], coords[goal])
    return 10.0  # fallback heuristic

def get_shared_buses(source: str, target: str) -> List[str]:
    source_buses = set(karachi_graph[source]["buses"])
    target_buses = set(karachi_graph[target]["buses"])
    return sorted(list(source_buses & target_buses))

def recovery_node(current: str, visited: set) -> Optional[Tuple[str, float]]:
    if current not in coords:
        return None
    current_coord = coords[current]
    candidate = None
    min_dist = float('inf')
    for node, node_coord in coords.items():
        if node == current or node in visited:
            continue
        d = euclidean_distance(current_coord, node_coord)
        if d < min_dist:
            min_dist = d
            candidate = node
    if candidate is not None:
        return candidate, min_dist
    return None

# --------------------------------
# Improved A* Search With Recovery
# --------------------------------

def a_star_search_with_recovery(start: str, goal: str, switch_penalty_value: float = 1.0):
    start = normalize_name(start)
    goal = normalize_name(goal)
    open_set = []
    heapq.heappush(open_set, (0, start, 0, [start], [], None))
    g_score = {start: 0}

    while open_set:
        f_val, current, current_g, path, bus_transitions, last_bus = heapq.heappop(open_set)

        if current == goal:
            return path, bus_transitions

        if current_g > g_score.get(current, float('inf')):
            continue

        neighbors = [normalize_name(n) for n in karachi_graph.get(current, {}).get("connected", []) if n in karachi_graph]

        if not neighbors:
            recovery = recovery_node(current, set(path))
            if recovery:
                rec_node, rec_cost = recovery
                tentative_g = current_g + rec_cost
                if tentative_g < g_score.get(rec_node, float('inf')):
                    g_score[rec_node] = tentative_g
                    h_score = heuristic(rec_node, goal)
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (
                        f_score, rec_node, tentative_g,
                        path + [rec_node],
                        bus_transitions + [(current, rec_node, "RECOVERY_JUMP")],
                        None
                    ))
            continue

        for neighbor in neighbors:
            shared_buses = get_shared_buses(current, neighbor)
            best_bus = shared_buses[0] if shared_buses else None

            switch_penalty = 0 if last_bus == best_bus or last_bus is None else switch_penalty_value
            tentative_g = current_g + 1 + switch_penalty

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                h_score = heuristic(neighbor, goal)
                f_score = tentative_g + h_score
                heapq.heappush(open_set, (
                    f_score, neighbor, tentative_g,
                    path + [neighbor],
                    bus_transitions + [(current, neighbor, best_bus)],
                    best_bus
                ))

    return None, None

# -----------------------------
# Breadth-First Search (BFS) with Recovery
# -----------------------------
def bfs_with_recovery(start: str, goal: str):
    queue = deque([(start, [start], [])])
    visited = {start}
    while queue:
        curr, path, trans = queue.popleft()
        if curr == goal:
            return path, trans
        if curr not in karachi_graph:
            continue
        neighbors = [
            n for n in karachi_graph[curr].get("connected", [])
            if n in karachi_graph and n not in visited
        ]
        if neighbors:
            for n in neighbors:
                visited.add(n)
                buses = get_shared_buses(curr, n)
                bus = buses[0] if buses else None
                queue.append((n, path + [n], trans + [(curr, n, bus)]))
        else:
            rec = recovery_node(curr, visited)
            if rec:
                node, _ = rec
                visited.add(node)
                queue.append((node, path + [node], trans + [(curr, node, "RECOVERY_JUMP")]))
    return None, []

# -----------------------------
# Evolutionary Algorithm (EA)
# -----------------------------
def evaluate_path(path: List[str], goal: str, switch_penalty: float = 10.0):
    total_cost = 0.0
    last_bus = None
    visited = {path[0]}
    trans = []
    for a, b in zip(path, path[1:]):
        if b in karachi_graph.get(a, {}).get("connected", []):
            if a in coords and b in coords:
                dist = euclidean_distance(coords[a], coords[b])
            else:
                dist = 1.0  # fallback if coordinates missing
            total_cost += dist
            buses = get_shared_buses(a, b)
            bus = buses[0] if buses else None
            if last_bus and bus and last_bus != bus:
                total_cost += switch_penalty
            trans.append((a, b, bus))
            last_bus = bus
        else:
            rec = recovery_node(a, visited)
            if rec and rec[0] == b:
                total_cost += rec[1]
                trans.append((a, b, "RECOVERY_JUMP"))
                last_bus = None
            else:
                total_cost += 1000
                trans.append((a, b, "INVALID"))
        visited.add(b)
    if path[-1] != goal:
        total_cost += 1000
    return total_cost, trans

def mutate_path(path: List[str]):
    if len(path) < 3:
        return path
    idx = random.randint(1, len(path) - 2)
    rec = recovery_node(path[idx], set(path))
    if rec:
        path[idx] = rec[0]
    return path

def crossover_path(p1: List[str], p2: List[str]):
    common = set(p1) & set(p2) - {p1[0], p1[-1]}
    if not common:
        return p1[:]
    pivot = random.choice(list(common))
    i1, i2 = p1.index(pivot), p2.index(pivot)
    return p1[:i1] + p2[i2:]

def evolutionary_search(start: str, goal: str, generations: int = 50, pop_size: int = 20):
    nodes = list(karachi_graph.keys())
    population = [[start] + random.sample(nodes, min(3, len(nodes))) + [goal]
                  for _ in range(pop_size)]
    for _ in range(generations):
        population.sort(key=lambda p: evaluate_path(p, goal)[0])
        survivors = population[:pop_size // 2]
        while len(survivors) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = crossover_path(p1, p2)
            if random.random() < 0.3:
                child = mutate_path(child)
            survivors.append(child)
        population = survivors
    best = min(population, key=lambda p: evaluate_path(p, goal)[0])
    cost, trans = evaluate_path(best, goal)
    return best, trans

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    start_node = "Tower"
    goal_node = "Numaish"

    # A* Search
    path_a, trans_a = a_star_search_with_recovery(start_node, goal_node, switch_penalty_value=1000.0)
    if path_a:
        print("\n--- A* Search ---")
        print(" -> ".join(path_a))
        print("\nDetailed transitions:")
        for frm, to, bus in trans_a:
            if bus == "RECOVERY_JUMP":
                print(f"From '{frm}' to '{to}' via recovery jump")
            elif bus:
                print(f"From '{frm}' to '{to}' using bus: {bus}")
            else:
                print(f"From '{frm}' to '{to}' (switch penalty applied)")
    else:
        print(f"No path found from {start_node} to {goal_node}.")

    # BFS Search
    path_bfs, trans_bfs = bfs_with_recovery(start_node, goal_node)
    if path_bfs:
        print("\n--- BFS Search ---")
        print(" -> ".join(path_bfs))
        print("\nDetailed transitions:")
        for frm, to, bus in trans_bfs:
            if bus == "RECOVERY_JUMP":
                print(f"From '{frm}' to '{to}' via recovery jump")
            elif bus:
                print(f"From '{frm}' to '{to}' using bus: {bus}")
            else:
                print(f"From '{frm}' to '{to}' (switch penalty applied)")
    else:
        print(f"No path found from {start_node} to {goal_node}.")

    # Evolutionary Algorithm
    path_ea, trans_ea = evolutionary_search(start_node, goal_node)
    if path_ea:
        print("\n--- Evolutionary Algorithm ---")
        print(" -> ".join(path_ea))
        print("\nDetailed transitions:")
        for frm, to, bus in trans_ea:
            if bus == "RECOVERY_JUMP":
                print(f"From '{frm}' to '{to}' via recovery jump")
            elif bus == "INVALID":
                print(f"Invalid move from '{frm}' to '{to}'")
            elif bus:
                print(f"From '{frm}' to '{to}' using bus: {bus}")
            else:
                print(f"From '{frm}' to '{to}' (switch penalty applied)")
    else:
        print(f"No path found from {start_node} to {goal_node}.")