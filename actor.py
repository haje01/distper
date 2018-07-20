"""액터 모듈."""
import os
import time
import pickle

import zmq
import numpy as np
import torch

from common import Experience, ExperienceBuffer, ENV_NAME, ActorInfo,\
    float2byte, byte2float, get_logger, DQN, async_recv
from wrappers import make_env

SHOW_FREQ = 100   # 로그 출력 주기
BUFFER_SIZE = 70  # 보낼 버퍼 크기
MODEL_UPDATE_FREQ = 300    # 러너의 모델 가져올 주기
EPS_BASE = 0.4   # eps 계산용
# EPS_ALPHA = 7    # eps 계산용
EPS_ALPHA = 1    # eps 계산용

actor_no = int(os.environ.get('ACTOR_NO', '-1'))   # 이 액터의 번호
assert actor_no != -1
num_actor = int(os.environ.get('NUM_ACTOR', '-1'))  # 전체 액터 수
assert num_actor != -1

log = get_logger()


def init_zmq():
    """ZMQ관련 초기화."""
    context = zmq.Context()

    # 러너에서 받을 소캣
    lrn_sock = context.socket(zmq.SUB)
    lrn_sock.setsockopt_string(zmq.SUBSCRIBE, '')
    lrn_sock.setsockopt(zmq.CONFLATE, 1)
    lrn_sock.connect("tcp://localhost:5557")

    # 버퍼로 보낼 소켓
    buf_sock = context.socket(zmq.PUSH)
    buf_sock.connect("tcp://localhost:5558")
    return context, lrn_sock, buf_sock


class Agent:
    """에이전트."""

    def __init__(self, env, exp_buffer, epsilon):
        """초기화."""
        self.env = env
        self.exp_buffer = exp_buffer
        self.epsilon = epsilon
        self._reset()

    def _reset(self):
        """리셋 구현."""
        self.state = float2byte(self.env.reset())
        self.tot_reward = 0.0

    def play_step(self, net, epsilon, frame_idx):
        """플레이 진행."""
        done_reward = None

        if np.random.random() < self.epsilon:
            # 임의 동작
            action = self.env.action_space.sample()
        else:
            # 가치가 높은 동작.
            state = byte2float(self.state)
            state_a = np.array([state])
            state_v = torch.tensor(state_a)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # 환경 진행
        new_state, reward, is_done, _ = self.env.step(action)
        new_state = float2byte(new_state)
        self.tot_reward += reward

        # 버퍼에 추가
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if frame_idx % SHOW_FREQ == 0:
            log("{}: buffer size {} ".format(frame_idx, len(self.exp_buffer)))

        # 종료되었으면 리셋
        if is_done:
            done_reward = self.tot_reward
            self._reset()

        # 에피소드 리워드 반환
        return done_reward

    @property
    def exp_full(self):
        """경험 버퍼가 다 찼는지 여부."""
        return len(self.exp_buffer) == BUFFER_SIZE

    def send_prioritized_replay(self, buf_sock, info):
        """우선화 리플레이를 버퍼에 보냄."""
        log("send replay - speed {} f/s".format(info.speed))
        # TODO: 우선화
        # 일단 그냥 다 보냄
        payload = pickle.dumps((actor_no, self.exp_buffer, info))
        buf_sock.send(payload)
        # 버퍼 클리어 (향후 우선화되면 필요없을 듯)
        self.exp_buffer.clear()


def receive_model(lrn_sock, net, block):
    """러너에게서 모델을 받음."""
    log("receive model from learner.")
    if block:
        param = lrn_sock.recv()
    else:
        param = async_recv(lrn_sock)

    if param is None:
        log("no new model. use old one.")
        return net

    log(net.state_dict()['conv.0.weight'][0][0])
    log("received new model.")
    state_dict = pickle.loads(param)
    net.load_state_dict(state_dict)
    log(net.state_dict()['conv.0.weight'][0][0])


def main():
    """메인."""
    # 환경 생성
    env = make_env(ENV_NAME)
    net = DQN(env.observation_space.shape, env.action_space.n)
    buffer = ExperienceBuffer(BUFFER_SIZE)
    # 고정 eps로 에이전트 생성
    epsilon = EPS_BASE ** (1 + actor_no / (num_actor - 1) * EPS_ALPHA)
    agent = Agent(env, buffer, epsilon)
    log("Actor {} - epsilon {:.5f}".format(actor_no, epsilon))

    # zmq 초기화
    context, lrn_sock, buf_sock = init_zmq()
    # 러너에게서 기본 가중치 받고 시작
    receive_model(lrn_sock, net, True)

    #
    # 시뮬레이션
    #
    episode = frame_idx = 0
    p_time = p_frame = None
    p_reward = -50.0

    while True:
        frame_idx += 1

        # 스텝 진행 (에피소드 종료면 reset까지)
        reward = agent.play_step(net, epsilon, frame_idx)

        # 리워드가 있는 경우 (에피소드 종료)
        if reward is not None:
            episode += 1
            p_reward = reward

        # 버퍼가 찼으면
        if agent.exp_full:
            # 학습관련 정보
            if p_time is None:
                speed = 0.0
            else:
                speed = (frame_idx - p_frame) / (time.time() - p_time)
            info = ActorInfo(episode, frame_idx, p_reward, speed)
            # 버퍼와 정보 전송
            agent.send_prioritized_replay(buf_sock, info)

            p_time = time.time()
            p_frame = frame_idx

        # 모델을 받을 때가 되었으면 받기
        if frame_idx % MODEL_UPDATE_FREQ == 0:
            receive_model(lrn_sock, net, False)


if __name__ == '__main__':
    main()
