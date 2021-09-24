import numpy as np
import torch
from numpy.random import default_rng
from torch import nn
from copy import deepcopy


def get_child_states_actions(env):
    next_state = []
    actions = env.legal_actions

    for a in actions:
        new_env = deepcopy(env)
        s, r, d, _ = new_env.step(a)

        next_state.append((new_env, s, a, d, r))

    return next_state


class Node:
    def __init__(self, env, state,
                 prev_action=None,
                 parent=None, 
                 p=1,
                 done=False,
                 result=0,
                 exploration_weight=1.0,
                 eta=0.03,
                 eps=0.25,
                 ):

        self.env = env
        self.state = state
        self.prev_action = prev_action
        self.parent = parent
        self.p = p
        self.exploration_weight = exploration_weight  # todo - should decay over search for evaluation games
        self.eta = eta
        self.eps = eps
        self.done = done
        self.result = result
        self.children = []

        self.v = 0
        self.n = 1
        self.w = 0
        self.q = 0
        if self.parent is not None:
            self.u = self.update_u()
        else:
            self.u = None

    def is_terminal(self):
        return self.done

    def max_qu_child(self):
        if len(self.children) == 0:
            return None

        qu = np.zeros(len(self.children))

        for i in range(len(self.children)):
            child = self.children[i]
            qu[i] = child.q + child.u

        i = qu.argmax()

        return self.children[i]

    # def to_torch(self):
    #     return torch.from_numpy(self.state).unsqueeze(0).to(dtype=torch.float32)

    def update_u(self):
        if self.parent is None:
            return None

        return self.exploration_weight * self.p * np.sqrt(self.parent.n) / (1 + self.n)

    def policy(self, A, temperature=1.0):

        policy = np.zeros(A)

        for child in self.children:
            policy[child.prev_action] = (child.n / self.n) ** (1 / temperature)

        scaled_policy = policy / policy.sum()  # todo - can we do this

        if np.isnan(scaled_policy).any():
            print("problem")

        return scaled_policy

    def expand_children(self, pi):
        if self.parent is None:
            # root node
            #  -- Add D Noise --
            #  P(s,a) = (1 − epsilon)*p_a+ epsilon*d_noise , where d_noise∼Dir(0.03) and epsilon=0.25
            alpha = np.full(pi.shape, self.eta)
            d_noise = np.random.dirichlet(alpha, size=None)  # numpy.random.dirichlet(alpha, size=None)
            # Apply D Noise
            pi = (1 - self.eps) * pi + self.eps * d_noise

        if not self.done:
            child_states_actions = get_child_states_actions(self.env)

            for child_env, state, action, child_done, child_return in child_states_actions:
                new_node = Node(
                    env=child_env,
                    state=state,
                    eps=self.eps,
                    eta=self.eta,
                    parent=self,
                    prev_action=action,
                    p=pi[action],
                    done=child_done,
                    result=child_return,
                )
                self.children.append(new_node)

    def child_by_action(self, a):
        if len(self.children) == 0:
            child_states_actions = get_child_states_actions(self.env)

            for child_env, state, action, child_done, child_return in child_states_actions:
                new_node = Node(
                    env=child_env,
                    state=state,
                    eps=self.eps,
                    eta=self.eta,
                    parent=self,
                    prev_action=action,
                    done=child_done,
                    result=child_return,
                )
                self.children.append(new_node)

        # returns the child corresponding to the given action
        for child in self.children:
            if child.prev_action == a:
                return child

        raise IndexError("Child for node with action {} not found", a)

class MCTS:
    # todo - currently we make a new mcts search tree every move. This is extremely inefficient. Investigate setting
    #  it up so that we reuse relevant parts of the tree every move - should let the tree grow much larger

    # todo - detect and link transpositions. For example, 1. e4 d4 2. e5 d5 leads to the same board position as 1. e5
    #  d5 2. e4 d4, so we should be able to link those two nodes together so we aren't processing the same position
    #  multiple times.
    def __init__(self,
                 game,
                 state,
                 rollouts,
                 A,
                 device,
                 net: nn.Module,
                 pi_temp=1,
                 eps=1.0,
                 eta=1.0,

                 ):
        self.game = game
        self.rollouts = rollouts
        self.A = A
        self.root: Node = Node(
                    env=game,
                    state=state,
                    eps=eps,
                    eta=eta,
                    p=None,
                    done=False,
                    result=None,
                )
        self.net = net
        self.pi_temp = pi_temp
        self.device = device
        self.rng = default_rng()

    def act(self):
        for i in range(self.rollouts):
            leaf = self.select(self.root)
            next_node, v = self.expand(leaf)
            self.back_up(next_node, v)

        pi = self.root.policy(self.game.action_space.n, temperature=self.pi_temp)

        a = self.rng.choice(np.arange(self.A), p=pi)
        a = int(a)

        # we now want to set the root of the tree to be the node corresponding to the chosen action
        self.root = self.root.child_by_action(a)
        self.root.parent = None

        return a, pi

    def select(self, current_node: Node):
        if len(current_node.children) == 0:
            return current_node
        else:
            return self.select(current_node.max_qu_child())

    def expand(self, leaf):
        if leaf.is_terminal():
            return leaf.parent, leaf.result

        else:
            net_pi, net_v = self.net(leaf.state)

            leaf.w = net_v

            leaf.expand_children(net_pi)

            return leaf.parent, -net_v

    def back_up(self, current_node: Node, v):
        if current_node is None:
            return

        current_node.n += 1
        current_node.w += v
        current_node.q = current_node.w / current_node.n

        current_node.update_u()

        self.back_up(current_node.parent, -v)

    def opponent_action(self, a):
        # we now want to set the root of the tree to be the node corresponding to the chosen action
        self.root = self.root.child_by_action(a)
        self.root.parent = None

