import numpy as np
import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children
    Used for priority-based sampling
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        """Update the sum tree when a priority changes"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find the index of a sample with priority s"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return the total priority in the tree"""
        return self.tree[0]

    def add(self, priority, data):
        """Add a new experience with priority p"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Update the priority of an experience"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get experience and index by priority amount s"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using a sum tree
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha  # Priority exponent (how much to prioritize)
        self.beta = beta_start  # Importance sampling exponent
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.epsilon = 1e-5  # Small constant to prevent zero priority
        self.max_priority = 1.0  # Max priority to use for new samples
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer with max priority"""
        experience = Transition(state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)
        
    def sample(self, batch_size, device):
        """Sample a batch of experiences based on priority"""
        indices = []
        priorities = []
        transitions = []
        segment = self.tree.total() / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample uniformly from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** -self.beta
        weights = weights / weights.max()  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Convert list of transitions to batch
        batch = Transition(*zip(*transitions))
        # Prepare batch for neural network input
        batch_state = torch.from_numpy(np.stack(batch.state)).to(device).squeeze(1)
        batch_action = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        batch_reward = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        batch_next_state = torch.from_numpy(np.stack(batch.next_state)).to(device).squeeze(1)
        batch_done = torch.tensor(batch.done, dtype=torch.bool, device=device)
        
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights
        
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.tree.n_entries