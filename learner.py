"""러너 모듈."""

import time
import pickle

import zmq
import torch
from torch import nn
from torch.nn import functional as F  # NOQA
from torch import optim
from tensorboardX import SummaryWriter

from common import DQN, ENV_NAME, get_device, byte2float, get_logger
from wrappers import make_env

STOP_REWARD = 19.5

GAMMA = 0.99
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000  # for GPU

GRADIENT_CLIP = 40

total_q_max = 0.0
log = get_logger()


def init_zmq():
    """ZMQ 초기화."""
    context = zmq.Context()

    # 액터로 보낼 소켓
    act_sock = context.socket(zmq.PUB)
    act_sock.bind("tcp://*:5557")

    # 버퍼에서 배치 받을 소켓
    buf_sock = context.socket(zmq.REQ)
    buf_sock.connect("tcp://localhost:5555")
    return context, act_sock, buf_sock


def calc_loss(batch, net, tgt_net, device):
    """손실 계산."""
    global total_q_max
    states, actions, rewards, dones, next_states = batch
    states = byte2float(states)
    next_states = byte2float(next_states)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    qs = net(states_v)
    total_q_max += float(qs[0].max())
    state_action_values = qs.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def publish_model(net, sync_net, act_sock):
    """가중치를 발행."""
    log("publish model.")
    cpu_model_dict = {}
    for key, val in net.state_dict().items():
        cpu_model_dict[key] = val.cpu()
    sync_net.load_state_dict(cpu_model_dict) 
    payload = pickle.dumps(sync_net)
    act_sock.send(payload)


def main():
    """메인 함수."""
    # 환경 생성
    env = make_env(ENV_NAME)
    device = get_device()
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())
    sync_net = DQN(env.observation_space.shape, env.action_space.n)
    writer = SummaryWriter(comment="-" + ENV_NAME)
    log(net)

    # ZMQ 초기화
    context, act_sock, buf_sock = init_zmq()
    log("Press Enter when the actors are ready: ")
    input()
    # 기본 모델을 발행해 액터 시작
    log("sending parameters to actors…")
    publish_model(net, sync_net, act_sock)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    fps = 0.0
    p_time = None
    train_cnt = 1
    while True:

        # 버퍼에게 학습을 위한 배치를 요청
        log("request new batch {}.".format(train_cnt))
        buf_sock.send(b'')
        payload = buf_sock.recv()

        if payload == b'not enough':
            # 아직 배치가 부족
            log("not enough data to train.")
            time.sleep(1)
        else:
            # 배치 학습
            log("train batch.")
            train_cnt += 1
            batch, ainfo, binfo = pickle.loads(payload)
            optimizer.zero_grad()
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()

            # Gradient Exploding에는 BN보다 Gradient Clip이 유효
            for param in net.parameters():
                param.grad.data.clamp_(-GRADIENT_CLIP, GRADIENT_CLIP)

            writer.add_scalar("learner/loss", float(loss_t), train_cnt)
            writer.add_scalar("learner/fps", fps, train_cnt)
            writer.add_scalar("learner/Qmax", float(total_q_max / ainfo.frame),
                              ainfo.frame)
            writer.add_scalar("buffer/replay", binfo.replay, train_cnt)
            writer.add_scalar("actor/fps", ainfo.speed, ainfo.frame)
            writer.add_scalar("actor/reward", ainfo.reward, ainfo.frame)
            optimizer.step()

            # 타겟 네트워크 갱신
            if train_cnt % SYNC_TARGET_FRAMES == 0:
                log("sync target network.")
                log(net.state_dict()['conv.0.weight'][0][0])
                tgt_net.load_state_dict(net.state_dict())

        # 모델 발행
        publish_model(net, sync_net, act_sock)
        if p_time is not None:
            elapsed = time.time() - p_time
            fps = 1.0 / elapsed
        p_time = time.time()

    writer.close()


if __name__ == '__main__':
    main()
    # Give 0MQ time to deliver
    time.sleep(1)
