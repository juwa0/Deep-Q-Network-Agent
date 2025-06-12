import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── DQN network ────────────────────────────
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1   = nn.Linear(32 * 9 * 9, 256)
        self.head  = nn.Linear(256, n_actions)

        # He init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.head(x)

    # ε-greedy helper
    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.head.out_features)
        with torch.no_grad():
            t = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
            return int(self(t).argmax(dim=1).item())

    # Save / load
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: torch.device | str = DEVICE):
        self.load_state_dict(torch.load(path, map_location=map_location))

# ──────────────────────── Replay Buffer ─────────────────────────────
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *tr: Transition):
        self.buf.append(tuple(tr))  # type: ignore[arg-type]

    def sample(self, batch: int):
        s, a, r, s2, d = map(np.array, zip(*random.sample(self.buf, batch)))
        t = lambda x, dtype: torch.as_tensor(x, device=DEVICE, dtype=dtype)
        return (t(s, torch.uint8), t(a, torch.int64), t(r, torch.float32),
                t(s2, torch.uint8), t(d, torch.bool))

    def __len__(self) -> int:
        return len(self.buf)

# ───────────────────── helper for fresh episodes ────────────────────
def empty_state():
    return np.zeros((4, 84, 84), dtype=np.uint8)

