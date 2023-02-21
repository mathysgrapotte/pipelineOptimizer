"""
Okay so here we are trying to implement a montecarlo tree search in its vanilla form, so to really understand how that works:

First we consider a start state, in our case that would be no parameter setup with the choice of parameter one two or three. 
Interestingly we could ask the tree to setup parameters in a different order right ? 
Although in terms of backprop, I am unsure on how that would work, that's an idea to keep for another time.




"""

# Monte Carlo Tree Search implementation

import numpy as np
import hashlib
import json
from collections import defaultdict

granularity = 200

parameters_range = {
    "p1":np.linspace(start=0, stop=1, num=granularity, dtype=float),
    "p2":np.linspace(start=0, stop=100, num=granularity, dtype=float),
    "p3":np.linspace(start=0, stop=50, num=granularity, dtype=float)
}

# Main loop 
def env_function(parameters, start):
    computed = (start ** parameters["p1"] + parameters["p2"]) * 2**parameters["p3"]
    # hard coded correct parameters
    correct = (start**0.47 + 38) * 2**12
    return computed, correct


class Node:

    # Node initialises with a state that is an empty dictionnary with keys p1, p2 and p3
    def __init__(self, node_state={"p1":None, "p2":None, "p3":None}, input=12, parameters_range=parameters_range):
        self.parameters_range = parameters_range
        self.node_state = node_state
        self.input = input
        self.visits = 0
        self.value = 0

    def is_terminal(self):
        return self.node_state["p1"] is not None and self.node_state["p2"] is not None and self.node_state["p3"] is not None

    def get_reward(self):
        assert self.is_terminal(), "Cannot get reward of a non terminal node"
        computed = env_function(self.node_state, self.input)[0]
        correct = env_function(self.node_state, self.input)[1]
        # Compute the distance between the computed and the correct value
        return 1/(1+abs(computed-correct))

    def get_children(self):
        children = []
        for parameter in self.node_state.keys():
            if self.node_state[parameter] is None:
                for value in self.parameters_range[parameter]:
                    new_state = self.node_state.copy()
                    new_state[parameter] = value
                    children.append(Node(new_state, self.input, self.parameters_range))
                break
        return children

    def get_random_child(self):
        return np.random.choice(self.get_children())


    def __hash__(self):
        # Hash the node state using the state dictionnary values
        dhash = hashlib.md5(json.dumps(self.node_state).encode('utf-8')).hexdigest()
        return int(dhash, 16)

class MCTS:
    def __init__(self, start_state=Node()):
        self.start_state = start_state
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.hash_node_table = defaultdict(dict)
        self.hash_reward_table = defaultdict(float)
        self.expanded_nodes_hashes = set()

    def iterate(self, iterations=10000):
        for i in range(iterations):
            # We sometimes print the results
            #if i % 1000 == 0:
            #    self.rollout(self.start_state, print_results=True)
            #else:
            self.rollout(self.start_state)

    def rollout(self, node, print_results=False):
        # Train for one iteration
        path = []
        while not node.is_terminal():
            # get the children of the node
            children = node.get_children()
            # build a set of hashes of the children 
            children_hashes = set([child.__hash__() for child in children])
            # If the node is fully extended, we get the best child
            if children_hashes.issubset(self.expanded_nodes_hashes):
                node = self.select_best_child(node, children)
                path.append(node)
            # If the node is not fully extended, we get an unseen child
            else:
                node = node.get_random_child()
                path.append(node)

        # We get the reward of the terminal node
        reward = node.get_reward()
        self.hash_reward_table[node.__hash__()] = reward
        if print_results:
            print(reward)
        
        # We backpropagate the reward
        self.backprop(path, reward)
        
    def uct(self, parent_node, child_node):
        return self.Q[child_node.__hash__()]/self.N[child_node.__hash__()] + np.sqrt(2*np.log(self.N[parent_node.__hash__()])/self.N[child_node.__hash__()])

    def select_best_child(self, node, children):
        return max(children, key=lambda child: self.uct(node, child))
   
    def backprop(self, path, reward):
        for node in path:
            self.hash_node_table[node.__hash__()] = node.node_state
            self.Q[node.__hash__()] += reward
            self.N[node.__hash__()] += 1
            self.expanded_nodes_hashes.add(node.__hash__())




    



