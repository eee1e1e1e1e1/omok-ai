import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """ MCTS 트리의 노드.
    Q : 자신의 값 
    P : 사전 확률(현재 정보를 기초로 정한 초기 확률 ex: 동전의 앞면이 나올 확률 ½)
    u : visit-count 사전 점수 
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, forbidden_moves, is_you_black):
        """Expansion  
        자식 노드를 생성하여 트리 확장 
        action_priors: 함수에 따라 사전확률과 액션들의 튜플의 리스트   """ 
        for action, prob in action_priors:
            # 흑돌일 때 금수 위치는 트리 탐색을 하지 않도록 
            if is_you_black and action in forbidden_moves : continue
            if action not in self._children : self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Selection 
        Q값과 u값을 더하여 자식노드에서 다음 액션을 선택 
        Return: 다음 노드의 튜플 
        """ 
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
       """leaf evaluation(자식노드)으로부터 노드 값을 업데이트 
        leaf_value: 현재 플레이어의 상황에서 서브트리의 값    """ 
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        # root가 아니면 , 반드시 이 노드의 부모노드가 먼저 생성되야함
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """노드의 값을 return 및 계산 
       leaf evaluations Q와 visit count u에 따라 사전 조정된 노드의 합.  
        c_puct: value Q의 값과 사전 확률 P가 이 노드의 점수에 미치는 영향을 통제하는 숫자.  """ 
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
       """leaf node 확인 ( 이 아래에 확장된 노드가 있는지)""" 
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf() : break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end : node.expand(action_probs, state.forbidden_moves, state.is_you_black())
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        # print([(state.move_to_location(m),v) for m,v in act_visits])

        # acts = 위치번호 / visits = 방문횟수
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move] # 돌을 둔 위치가 root노드가 됨
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(board.width*board.height)
        if board.width*board.height - len(board.states) > 0:
            # acts와 probs에 의해 착수 위치가 정해진다.
            acts, probs = self.mcts.get_move_probs(board, temp)      
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # (자가 학습을 할 때는) Dirichlet 노이즈를 추가하여 탐색
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs) # 확률론적인 방법
                self.mcts.update_with_move(-1)

            if return_prob : return move, move_probs
            else : return move
        
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
