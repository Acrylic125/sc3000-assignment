import json
import heapq



class Q1:
    def __init__(self):
        with open('Coord.json', 'r') as f:
            self.coord = json.load(f)
        with open('Cost.json', 'r') as f:
            self.cost = json.load(f)
        with open('Dist.json', 'r') as f:
            self.dist = json.load(f)
        with open('G.json', 'r') as f:
            self.G = json.load(f)
    
    def task_1(self):
        # Uniform cost search to from 1 to 50.
        # Task 1 says to ignore the energy constraint.
        # So we go by distance (Original problem)
        # (Distance, Path)[]
        goal = '50'
        parents = {}
        explored = set()
        best_dist = {}
        queue = [(0, '1')]
        heapq.heapify(queue)
        while len(queue) > 0:
            dist, node = heapq.heappop(queue)
            if node == goal:
                # Reconstruct path 
                path = []
                while node in parents:
                    path.append(node)
                    node = parents[node]
                path.append(node)
                path.reverse()
                print(f"Shortest distance from 1 to 50: {dist} {path}")
                return
            if node in explored:
                continue
            explored.add(node)
            for neighbor in self.G[node]:
                if neighbor not in explored:
                    edge_key = f"{min(node, neighbor)},{max(node, neighbor)}"
                    dist_neighbor = self.dist[edge_key]
                    updated_dist = dist + dist_neighbor
                    if best_dist.get(neighbor) != None and best_dist[neighbor] <= updated_dist:
                        continue
                    parents[neighbor] = node
                    best_dist[neighbor] = updated_dist
                    heapq.heappush(queue, (updated_dist, neighbor))
    
    def task_2(self):
        # Uniform cost search to from 1 to 50, with energy constraint.
        # (Distance, (Node, Remaining Energy))[]
        # Just like task 1, we prioritise distance. Energy constraint is a 
        # budget, not the main optimization.
        # Reuse UCS from Task 1, we reject any solutions that exceed the energy constraint.
        START_ENERGY = 287932
        start = "1"
        goal = '50'
        # There may be multiple possible paths to node due to energy.
        # f'{node},{energy}' -> (parent_node, parent_energy)
        parents = {}
        # We will use distance_and_energy as a way to prune redundant paths.
        # node -> (best_distance, best_energy)[]
        distance_and_energy = {
            start: [(0, START_ENERGY)] 
        }
        queue = [(0, ('1', START_ENERGY))]
        heapq.heapify(queue)
        while len(queue) > 0:
            dist, entry = heapq.heappop(queue)
            node, remaining_energy = entry
            if node == goal:
                # Reconstruct path 
                path = []
                cur_key = f"{node},{remaining_energy}"
                path.append(node)
                while cur_key in parents:
                    node, energy = parents[cur_key]
                    parent_key = f"{node},{energy}"
                    path.append(node)
                    cur_key = parent_key
                path.reverse()
                print(f"Shortest distance from 1 to 50: {dist} {path} with remaining energy, {remaining_energy}")
                return
            for neighbor in self.G[node]:
                edge_key = f"{min(node, neighbor)},{max(node, neighbor)}"
                dist_neighbor = self.dist[edge_key]
                updated_dist = dist + dist_neighbor
                updated_energy = remaining_energy - self.cost[edge_key]

                # Don't bother, we are already exhausted.
                if updated_energy < 0:
                    continue

                should_ignore = False 
                for best_dist, best_energy in distance_and_energy.get(neighbor, []):
                    # Shorter distance means theres still a possibility that this path is better, even if it has less energy. So we keep it.
                    # Larger energy means it is less exhausted, so it is possible that we have to resort to it later. So we keep it.
                    if best_dist <= updated_dist and best_energy >= updated_energy:
                        should_ignore = True
                        break
                if should_ignore:
                    continue

                neighbour_energy_key = f"{neighbor},{updated_energy}"
                # We need this to recover the path later.
                parents[neighbour_energy_key] = (node, remaining_energy)
                distance_and_energy[neighbor] = distance_and_energy.get(neighbor, []) + [(updated_dist, updated_energy)]
                heapq.heappush(queue, (updated_dist, (neighbor, updated_energy)))


if __name__ == '__main__':
    q1 = Q1()
    q1.task_1()
    q1.task_2()
