import json
import heapq
import random

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
        goal = '50'
        parents = {}
        explored = set()
        best_dist = {}
        # (Distance, Path)[]
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
                print(f'Shortest Path: {"->".join(path)}')
                print(f'Shortest Distance: {dist}')
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
        # Just like task 1, we prioritise distance. Energy constraint is a 
        # budget, not the main optimization.
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
        # (Distance, (Node, Remaining Energy))[]
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
                print(f'Shortest Path: {"->".join(path)}')
                print(f'Shortest Distance: {dist}')
                print(f'Total Energy Cost: {START_ENERGY - remaining_energy}')
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

    def task_3(self):
        # f(node) = g(node) + h(node)
        # Generally, a shorter path to our end goal uses less energy.
        # Let g(node) = current distance of node.
        # Let h(node) = distance(node, goal)
        # Note: We dont have to sqrt the distance.

        START_ENERGY = 287932
        start = "1"
        goal = '50'
        
        def h(node):
            node_coords = self.coord[node]
            goal_coords = self.coord[goal]
            return ((node_coords[0] - goal_coords[0]) ** 2 + (node_coords[1] - goal_coords[1]) ** 2)

        def f(node, dist):
            return dist + h(node)

        # There may be multiple possible paths to node due to energy.
        # f'{node},{energy}' -> (parent_node, parent_energy)
        parents = {}
        # We will use f_score_and_energy as a way to prune redundant paths.
        # node -> (f_score, best_energy)[]
        f_score_and_energy = {
            start: [(0, START_ENERGY)] 
        }
        # (F Score, (Node, Accumulated Distance, Remaining Energy))[]
        queue = [(f(start, 0), ('1', 0, START_ENERGY))]
        heapq.heapify(queue)
        while len(queue) > 0:
            f_score, entry = heapq.heappop(queue)
            node, dist, remaining_energy = entry
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
                print(f'Shortest Path: {"->".join(path)}')
                print(f'Shortest Distance: {dist}')
                print(f'Total Energy Cost: {START_ENERGY - remaining_energy}')
                return
            for neighbor in self.G[node]:
                edge_key = f"{min(node, neighbor)},{max(node, neighbor)}"
                dist_neighbor = self.dist[edge_key]
                updated_dist = dist + dist_neighbor
                updated_f_score = f(neighbor, updated_dist)
                updated_energy = remaining_energy - self.cost[edge_key]

                # Don't bother, we are already exhausted.
                if updated_energy < 0:
                    continue

                should_ignore = False 
                for best_f_score, best_energy in f_score_and_energy.get(neighbor, []):
                    # Shorter distance means theres still a possibility that this path is better, even if it has less energy. So we keep it.
                    # Larger energy means it is less exhausted, so it is possible that we have to resort to it later. So we keep it.
                    if best_f_score <= updated_f_score and best_energy >= updated_energy:
                        should_ignore = True
                        break
                if should_ignore:
                    continue

                neighbour_energy_key = f"{neighbor},{updated_energy}"
                # We need this to recover the path later.
                parents[neighbour_energy_key] = (node, remaining_energy)
                f_score_and_energy[neighbor] = f_score_and_energy.get(neighbor, []) + [(updated_f_score, updated_energy)]
                heapq.heappush(queue, (updated_f_score, (neighbor, updated_dist, updated_energy)))


CELL_EMPTY = 0
CELL_START = 1
CELL_GOAL = 2
CELL_BLOCK = 3

ACTION_LEFT = (0, -1, "<")
ACTION_UP = (1, 0, "↓") # Note the map is inverted, so we use down even though it's really up.
ACTION_RIGHT = (0, 1, ">")
ACTION_DOWN = (-1, 0, "↑") # Same thing here.

# Make sure the order is this so we can easily derive perp direcitons.
ACTIONS = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN]

class Q2:
    def __init__(self):
        self.grid = [
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_START, CELL_BLOCK, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_BLOCK, CELL_BLOCK, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_BLOCK, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_BLOCK, CELL_GOAL, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
            [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
        ]
        # self.grid = [
        #     [CELL_START, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
        #     [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
        #     [CELL_EMPTY, CELL_BLOCK, CELL_EMPTY, CELL_BLOCK, CELL_EMPTY],
        #     [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
        #     [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_GOAL],
        # ]

    def task_1_value_iteration(self):
        rows = len(self.grid)
        cols = len(self.grid[0]) 
        
        actions = ACTIONS
        
        intended_action_prob = 0.8
        unintended_action_prob = 0.1
        iterations = 100
        discounted_rate = 0.9

        max_change_threshold = 0.01

        V_old = [[0 for _ in range(cols)] for _ in range(rows)]
        policy_old = [[None for _ in range(cols)] for _ in range(rows)]
        for iteration in range(iterations):
            V_new = [[0 for _ in range(cols)] for _ in range(rows)]
            policy_new = [[None for _ in range(cols)] for _ in range(rows)]
            
            max_change = 0
            for row in range(rows):
                for col in range(cols):
                    # Skip terminal (goal) and invalid (block) states
                    if self.grid[row][col] == CELL_GOAL or self.grid[row][col] == CELL_BLOCK:
                        continue

                    best_Q_sa_and_action = None
                    for action_i, (_dx, _dy, symbol) in enumerate(actions):
                        # (dx, dy, Intended?)
                        perp_left_action = actions[(action_i + 1) % len(actions)]
                        perp_right_action = actions[(action_i - 1) % len(actions)]
                        dirs_to_consider = [
                            # Assume taking this action was intended
                            (_dx, _dy, True),
                            # But we also have to consider the unintended action.
                            (perp_left_action[0], perp_left_action[1], False),
                            (perp_right_action[0], perp_right_action[1], False),
                        ]

                        Q_sa = 0
                        for dx, dy, intended in dirs_to_consider:
                            new_row, new_col = row + dx, col + dy
                            # Out of bounds or wall: agent stays in place, still pays -1
                            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols or self.grid[new_row][new_col] == CELL_BLOCK:
                                intention_prob = intended_action_prob if intended else unintended_action_prob
                                Q_sa += intention_prob * (-1 + discounted_rate * V_old[row][col])
                                continue

                            reward = -1
                            if self.grid[new_row][new_col] == CELL_EMPTY:
                                reward = -1
                            if self.grid[new_row][new_col] == CELL_GOAL:
                                reward = 10

                            intention_prob = intended_action_prob if intended else unintended_action_prob
                            Q_sa += intention_prob * (reward + discounted_rate * V_old[new_row][new_col])

                        # Then we determine if this is the best action.
                        if best_Q_sa_and_action == None or Q_sa > best_Q_sa_and_action[0]:
                            best_Q_sa_and_action = (Q_sa, action_i)

                    if best_Q_sa_and_action != None:
                        Q_sa, action_i = best_Q_sa_and_action
                        V_new[row][col] = Q_sa
                        policy_new[row][col] = action_i
                        max_change = max(max_change, abs(V_new[row][col] - V_old[row][col]))

            V_old = V_new
            policy_old = policy_new
            if max_change < max_change_threshold:
                print(f'Converged after {iteration} iterations.')
                break

        # Show V and Policy
        for row in range(rows):
            for col in range(cols):
                print(f'{V_old[row][col]:.2f}', end=' ')
            print()

        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_BLOCK:
                    c = "#"
                elif self.grid[row][col] == CELL_GOAL:
                    c = "G"
                else:
                    c = actions[policy_old[row][col]][2]
                print(f'{c}', end=' ')
            print()

    def task_1_policy_iteration(self):
        rows = len(self.grid)
        cols = len(self.grid[0]) 
        
        actions = ACTIONS
        
        intended_action_prob = 0.8
        unintended_action_prob = 0.1
        iterations = 1000
        discounted_rate = 0.9

        max_change_threshold = 0.01 # omega

        # Initialize random policy. Doesnt matter.
        policy = [[0 for _ in range(cols)] for _ in range(rows)]
        V = [[0 for _ in range(cols)] for _ in range(rows)]

        for iteration in range(iterations):
            # Policy Evalation
            while True:
                max_change = 0
                for row in range(rows):
                    for col in range(cols):
                        if self.grid[row][col] == CELL_GOAL or self.grid[row][col] == CELL_BLOCK:
                            continue

                        action_i = policy[row][col]
                        _dx, _dy, symbol = actions[action_i]
                        perp_left_action = actions[(action_i + 1) % len(actions)]
                        perp_right_action = actions[(action_i - 1) % len(actions)]
                        dirs_to_consider = [
                            (_dx, _dy, True),
                            (perp_left_action[0], perp_left_action[1], False),
                            (perp_right_action[0], perp_right_action[1], False),
                        ]

                        v = 0
                        for dx, dy, intended in dirs_to_consider:
                            new_row, new_col = row + dx, col + dy
                            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols or self.grid[new_row][new_col] == CELL_BLOCK:
                                intention_prob = intended_action_prob if intended else unintended_action_prob
                                v += intention_prob * (-1 + discounted_rate * V[row][col])
                                continue

                            reward = -1
                            if self.grid[new_row][new_col] == CELL_EMPTY:
                                reward = -1
                            if self.grid[new_row][new_col] == CELL_GOAL:
                                reward = 10

                            intention_prob = intended_action_prob if intended else unintended_action_prob
                            v += intention_prob * (reward + discounted_rate * V[new_row][new_col])

                        prev_v = V[row][col]
                        V[row][col] = v
                        max_change = max(max_change, abs(prev_v - v))
                if max_change < max_change_threshold:
                    break
        
            # Policy Improvement.
            is_stable = True
            for row in range(rows):
                for col in range(cols):
                    current_action = policy[row][col]

                    # Find the best action.
                    best = None
                    for action_i, (_dx, _dy, symbol) in enumerate(actions):
                        perp_left_action = actions[(action_i + 1) % len(actions)]
                        perp_right_action = actions[(action_i - 1) % len(actions)]
                        dirs_to_consider = [
                            (_dx, _dy, True),
                            (perp_left_action[0], perp_left_action[1], False),
                            (perp_right_action[0], perp_right_action[1], False),
                        ]

                        Q_sa = 0
                        for dx, dy, intended in dirs_to_consider:
                            new_row, new_col = row + dx, col + dy
                            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols or self.grid[new_row][new_col] == CELL_BLOCK:
                                intention_prob = intended_action_prob if intended else unintended_action_prob
                                Q_sa += intention_prob * (-1 + discounted_rate * V[row][col])
                                continue

                            reward = -1
                            if self.grid[new_row][new_col] == CELL_EMPTY:
                                reward = -1
                            if self.grid[new_row][new_col] == CELL_GOAL:
                                reward = 10

                            intention_prob = intended_action_prob if intended else unintended_action_prob
                            Q_sa += intention_prob * (reward + discounted_rate * V[new_row][new_col])

                        if best == None or best[0] < Q_sa:
                            best = (Q_sa, action_i)
                            policy[row][col] = action_i

                    assert best != None, "There should be at least 1 valid action."
                    best_Q_sa, best_action_i = best
                    if current_action != best_action_i:
                        is_stable = False
            if is_stable:
                print(f'Converged after {iteration} iterations.')
                break
        # Show V and Policy
        for row in range(rows):
            for col in range(cols):
                print(f'{V[row][col]:.2f}', end=' ')
            print()

        for row in range(rows):
            for col in range(cols):
                c = actions[policy[row][col]][2]
                if self.grid[row][col] == CELL_BLOCK:
                    c = "#"
                # if self.grid[row][col] == CELL_START:
                #     c = "S"
                if self.grid[row][col] == CELL_GOAL:
                    c = "G"
                print(f'{c}', end=' ')
            print()
    
    def task_2_monte_carlo_control(self):
        # For consistent results.
        random.seed(42)
        rows = len(self.grid)
        cols = len(self.grid[0]) 
        
        actions = ACTIONS
        episodes = 1_000_000
        discounted_rate = 0.9
        intended_action_prob = 0.8
        unintended_action_prob = 0.1
        
        start = (0, 0)
        # Find the start state.
        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_START:
                    start = (row, col)

        # Initialize random policy. Doesnt matter.
        policy = [[random.randint(0, len(actions) - 1) for _ in range(cols)] for _ in range(rows)]
        # We will use returns to estimate the value of a state.
        # (State) -> [(Total, Count)] for each action.
        # NOTE: We will use returns to keep track of total reward and count
        # to derive the average then update the policy as we go, rather than
        # keeping track of each return. This is only a programmatic change, 
        # the logic is still Monte Carlo Control.
        returns = {}
        
        # Epsilon greedy policy.
        epsilon = 0.1

        for episode in range(episodes):
            # Play the episode.
            # (x, y, Action, Reward)[]
            episode_state_action_rewards = []
            cur = start
            max_steps = rows * cols * 20
            for _step in range(max_steps):
                x, y = cur
                if self.grid[x][y] == CELL_GOAL:
                    break

                action_i = policy[x][y]
                if random.random() < epsilon:
                    # Each action has a e / |A| chance of being chosen.
                    # Note: The policy action still has a 1 - e + (e / |A|) chance of being chosen.
                    action_i = random.randint(0, len(actions) - 1)

                _dx, _dy, _symbol = actions[action_i]
                perp_left_action = actions[(action_i + 1) % len(actions)]
                perp_right_action = actions[(action_i - 1) % len(actions)]

                r = random.random()
                if r < intended_action_prob:
                    dx, dy = _dx, _dy
                elif r < intended_action_prob + unintended_action_prob:
                    dx, dy = perp_left_action[0], perp_left_action[1]
                else:
                    dx, dy = perp_right_action[0], perp_right_action[1]

                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= rows or new_y < 0 or new_y >= cols or self.grid[new_x][new_y] == CELL_BLOCK:
                    new_x, new_y = x, y

                reward = 10 if self.grid[new_x][new_y] == CELL_GOAL else -1
                done = self.grid[new_x][new_y] == CELL_GOAL
                episode_state_action_rewards.append((x, y, action_i, reward))
                cur = (new_x, new_y)
                if done:
                    break
                
            # Update the returns for each state-action pair.
            # Consider only first visit
            visited = set()
            Gt = 0
            for x, y, action_i, reward in reversed(episode_state_action_rewards):
                state_action_key = (x, y, action_i)
                # Revisiting the same state-action pair should affect the return.
                Gt = reward + discounted_rate * Gt
                if state_action_key in visited:
                    continue
                visited.add(state_action_key)
                state_key = (x, y)
                if state_key not in returns:
                    returns[state_key] = [(0.0, 0) for _ in range(len(actions))]
                total, count = returns[state_key][action_i]
                returns[state_key][action_i] = (total + Gt, count + 1)
                
                # Update policy
                best = None
                for action_i in range(len(actions)):
                    total, count = returns[state_key][action_i]
                    if count == 0:
                        continue
                    value = total / count
                    if best == None or value > best[0]:
                        best = (value, action_i)
                if best != None:
                    policy[x][y] = best[1]
    
        # Show policy
        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_BLOCK:
                    c = "#"
                elif self.grid[row][col] == CELL_GOAL:
                    c = "G"
                else:
                    c = actions[policy[row][col]][2]
                print(f'{c}', end=' ')
            print()
    
    def task_3_q_learning(self):
        # For consistent results.
        random.seed(42)
        rows = len(self.grid)
        cols = len(self.grid[0]) 
        
        actions = ACTIONS
        episodes = 1_000_000
        discounted_rate = 0.9
        intended_action_prob = 0.8
        unintended_action_prob = 0.1
        learning_rate = 0.1 # alpha
        epsilon = 0.1
        
        start = (0, 0)
        # Find the start state.
        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_START:
                    start = (row, col)

        # Q[state] = list of Q(s, a) for each action
        Q = {}
        max_steps = rows * cols * 20

        for episode in range(episodes):
            cur = start
            for _ in range(max_steps):
                x, y = cur
                if self.grid[x][y] == CELL_GOAL:
                    break

                state = (x, y)
                if state not in Q:
                    Q[state] = [0.0] * len(actions)

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action_i = random.randint(0, len(actions) - 1)
                else:
                    action_i = max(range(len(actions)), key=lambda a: Q[state][a])

                _dx, _dy, _symbol = actions[action_i]
                perp_left_action = actions[(action_i + 1) % len(actions)]
                perp_right_action = actions[(action_i - 1) % len(actions)]

                r = random.random()
                if r < intended_action_prob:
                    dx, dy = _dx, _dy
                elif r < intended_action_prob + unintended_action_prob:
                    dx, dy = perp_left_action[0], perp_left_action[1]
                else:
                    dx, dy = perp_right_action[0], perp_right_action[1]

                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= rows or new_y < 0 or new_y >= cols or self.grid[new_x][new_y] == CELL_BLOCK:
                    new_x, new_y = x, y

                reward = 10 if self.grid[new_x][new_y] == CELL_GOAL else -1
                done = self.grid[new_x][new_y] == CELL_GOAL
                next_state = (new_x, new_y)

                if next_state not in Q:
                    Q[next_state] = [0.0] * len(actions)
                max_next_q = max(Q[next_state]) if not done else 0.0
                target = reward + discounted_rate * max_next_q
                Q[state][action_i] += learning_rate * (target - Q[state][action_i])

                cur = next_state
                if done:
                    break

        # Derive policy from Q: pi(s) = argmax_a Q(s, a)
        policy = [[0 for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_BLOCK or self.grid[row][col] == CELL_GOAL:
                    continue
                state = (row, col)
                q_list = Q.get(state, [0.0] * len(actions))
                policy[row][col] = max(range(len(actions)), key=lambda a: q_list[a])

        # Show policy
        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == CELL_BLOCK:
                    c = "#"
                elif self.grid[row][col] == CELL_GOAL:
                    c = "G"
                else:
                    c = actions[policy[row][col]][2]
                print(f'{c}', end=' ')
            print()

if __name__ == '__main__':
    q1 = Q1()
    q1.task_1()
    q1.task_2()
    q1.task_3()

    q2 = Q2()
    q2.task_1_value_iteration()
    q2.task_1_policy_iteration()
    q2.task_2_monte_carlo_control()
    q2.task_3_q_learning()
