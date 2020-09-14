import time
from chainer import cuda
from chainerrl import replay_buffer
import collections
import numpy as np


class EVAReplayBuffer(replay_buffer.ReplayBuffer):
    def __init__(self, capacity, num_steps, key_width, xp, M=10, T=50):
        super().__init__(capacity, num_steps)
        self.neighbors = M
        self.T = T
        self.key_width = key_width
        self.last_n_transitions = collections.defaultdict(
              lambda: collections.deque([], maxlen=num_steps))
        # self.last_n_transitions = collections.deque([], maxlen=num_steps)
        self.xp = xp
        self.correct_embeddings = []
        self.embeddings = self.xp.empty((0, key_width), dtype=self.xp.float32)

    def append(self, state, action, reward, embedding, next_state=None, next_action=None,
               is_state_terminal=False, env_id=0, **kwargs):
        last_n_transitions = self.last_n_transitions[env_id]

        experience = dict(
            state=state,
            action=action,
            reward=reward,
            embedding=embedding,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        # self.last_n_transitions.append(experience)
        last_n_transitions.append(experience)

        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                self.correct_embeddings.append(last_n_transitions[0]['embedding'].reshape(1, -1))
                del last_n_transitions[0]

            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))
                self.correct_embeddings.append(last_n_transitions[0]['embedding'].reshape(1, -1))

    def stop_current_episode(self, env_id=0):
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        last_n_transitions = self.last_n_transitions[env_id]
        if 0 < len(last_n_transitions) < self.num_steps:
            self.memory.append(list(last_n_transitions))
            # self.correct_embeddings.append(self.last_n_transitions[0]['embedding'].reshape(1, -1))
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del self.last_n_transitions[0]
        while last_n_transitions:
            self.memory.append(list(last_n_transitions))
            # self.correct_embeddings.append(self.last_n_transitions[0]['embedding'].reshape(1, -1))
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0

    def lookup(self, embedding):
        """lookup:Return trajectories from embedding by KNN.
        """
        # Get index of experiences gotten by index_serach
        # e.g.: [index of experience1, index of experience3, ...]
        I = cuda.to_cpu(self.index_search(embedding))
        # Extract the corresponding index experience and its subsequence (trajectory cut with length T)
        # e.g. : [[trajectory from experience1], [trajectory from experience3], ...]
        trajectories = [[self.memory[index + i] for i in range(min(self.T, len(self) - index))] for index in I]
        return trajectories

    def index_search(self, embedding):
        distances = self.xp.sum((self.xp.asarray(embedding) - self.embeddings) ** 2, axis=1)
        return self.xp.argsort(distances)[:self.neighbors]

    def update_embedding(self):
        if len(self.correct_embeddings) == 0:
            return 0
        embeddings = self.xp.asarray(np.concatenate(self.correct_embeddings, axis=0))
        self.embeddings = self.xp.concatenate((self.embeddings, embeddings), axis=0)[-self.capacity:]
        self.correct_embeddings = []