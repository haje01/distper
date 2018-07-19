"""공용 상수 및 함수."""
import sys
from collections import deque, namedtuple

import zmq
from torch.nn import functional as F  # NOQA
import numpy as np
import torch
from torch import nn

ENV_NAME = "PongNoFrameskip-v4"

ACTOR_BATCH_SIZE = 70       # 액터가 보낼 배치 크기(전이 수)
TRAIN_IMAGE_SIZE = 84       # 학습 이미지 크기

NO_OP_STEP = 30
EXPLORE_STEPS = 1000000
CLIP_TOP = 32
CLIP_BOTTOM = 18


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward',
                        'done', 'new_state'])


ActorInfo = namedtuple('ActorInfo',
                       field_names=['episode', 'frame', 'reward',
                                    'speed'])


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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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

    def __init__(self, capacity):
        """초기화."""
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """길이 연산자."""
        return len(self.buffer)

    def append(self, experience):
        """경험 추가."""
        self.buffer.append(experience)

    def merge(self, other):
        """다른 버퍼 내용을 병합."""
        self.buffer += other.buffer

    def sample(self, batch_size):
        """경험 샘플링."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states =\
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

    def clear(self):
        """버퍼 초기화."""
        self.buffer.clear()


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
