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
        queue = [(0, ['1'])]
        heapq.heapify(queue)
        goal = '50'
        explored = set()
        while len(queue) > 0:
            dist, path = heapq.heappop(queue)
            node = path[-1] # latest node
            if node == goal:
                print(f"Shortest distance from 1 to 50: {dist} {path}")
                return
            if node in explored:
                continue
            explored.add(node)
            for neighbor in self.G[node]:
                if neighbor not in explored:
                    dist_key = f"{min(node, neighbor)},{max(node, neighbor)}"
                    dist_neighbor = self.dist[dist_key]
                    heapq.heappush(queue, (dist + dist_neighbor, [*path, neighbor]))

if __name__ == '__main__':
    q1 = Q1()
    q1.task_1()
