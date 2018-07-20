"""공용 상수 및 함수."""
import sys
import logging
import random
from collections import namedtuple

import zmq
from torch.nn import functional as F  # NOQA
import numpy as np
import torch
from torch import nn

from sumtree import SumTree

ENV_NAME = "PongNoFrameskip-v4"

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward',
                        'done', 'new_state'])


ActorInfo = namedtuple('ActorInfo',
                       field_names=['episode', 'frame', 'reward', 'speed'])

BufferInfo = namedtuple('BufferInfo', field_names=['replay'])


def async_recv(sock):
    """비동기로 받음."""
    try:
        return sock.recv(zmq.DONTWAIT)
    except zmq.Again:
        pass


def get_device():
    """PyTorch에서 사용할 디바이스 얻음."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device.".format(dev.upper()))
    device = torch.device(dev)
    return device


class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, input_shape, n_actions):
        """초기화."""
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """전방 연쇄."""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ExperienceBuffer:
    """경험 버퍼."""

    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        """초기화."""
        self.tree = SumTree(capacity)
        # self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """길이 연산자."""
        return self.tree.n_entries

    def get_priority(self, error):
        """우선도 계산."""
        return (error + self.e) ** self.a

    def append(self, error, sample):
        """경험 추가."""
        p = self._get_priority(error)
        self.tree.add(p, sample)
        # self.buffer.append(experience)

    def sample(self, n):
        """우선 샘플링."""
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities,
                             -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        """우선 트리 갱신."""
        p = self._get_priority(error)
        self.tree.update(idx, p)

    # def merge(self, other):
    #     """다른 버퍼 내용을 병합."""
    #     self.buffer += other.buffer

    # def sample(self, batch_size):
    #     """경험 샘플링."""
    #     indices = np.random.choice(len(self.buffer), batch_size,
    # replace=False)
    #     states, actions, rewards, dones, next_states =\
    #         zip(*[self.buffer[idx] for idx in indices])
    #     return np.array(states), np.array(actions), \
    #         np.array(rewards, dtype=np.float32), \
    #         np.array(dones, dtype=np.uint8), np.array(next_states)

    # def clear(self):
    #     """버퍼 초기화."""
    #     self.buffer.clear()


def get_size(obj, seen=None):
    """Recursively finds size of objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def float2byte(data):
    """Float 이미지를 byte 이미지로."""
    return np.uint8(data * 255)


def byte2float(data):
    """Byte 이미지를 float 이미지로."""
    return np.float32(data / 255.0)


def get_logger():
    """로거 얻기."""
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()
    return logger.info
