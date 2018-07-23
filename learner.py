"""러너 모듈."""

import time
import pickle
from io import BytesIO

import zmq
import numpy as np
import torch
from torch.nn import functional as F  # NOQA
from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from common import DQN, ENV_NAME, get_device, get_logger, calc_loss,\
    weights_init, Experience
from wrappers import make_env

STOP_REWARD = 19.5

LEARNING_RATE = 1e-5
SYNC_TARGET_FREQ = 10

GRADIENT_CLIP = 40

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


def publish_model(net, tgt_net, act_sock):
    """가중치를 발행."""
    log("publish model.")
    bio = BytesIO()
    torch.save(net, bio)
    torch.save(tgt_net, bio)
    act_sock.send(bio.getvalue())
    # cpu_model_state = {}
    # for key, val in net.state_dict().items():
    #     cpu_model_state[key] = val.cpu()
    # payload = pickle.dumps(cpu_model_state)
    # act_sock.send(payload)


def main():
    """메인 함수."""
    # 환경 생성
    env = make_env(ENV_NAME)
    device = get_device()
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.apply(weights_init)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())
    writer = SummaryWriter(comment="-" + ENV_NAME)
    log(net)

    # ZMQ 초기화
    context, act_sock, buf_sock = init_zmq()
    log("Press Enter when the actors are ready: ")
    input()
    # 기본 모델을 발행해 액터 시작
    log("sending parameters to actors…")
    publish_model(net, tgt_net, act_sock)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
    #                             momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    fps = total_q_max = 0.0
    p_time = idxs = errors = None
    train_cnt = 1
    while True:

        # 버퍼에게 학습을 위한 배치를 요청
        log("request new batch {}.".format(train_cnt))
        payload = pickle.dumps((idxs, errors))
        buf_sock.send(payload)
        payload = buf_sock.recv()

        if payload == b'not enough':
            # 아직 배치가 부족
            log("not enough data to batch.")
            time.sleep(1)
        else:
            # 배치 학습
            log("train batch.")
            train_cnt += 1

            exps, idxs, ainfos, binfo = pickle.loads(payload)
            batch = Experience(*map(np.concatenate, zip(*exps)))
            loss_t, errors, q_maxs = calc_loss(batch, net, tgt_net,
                                               device=device)
            optimizer.zero_grad()
            loss_t.backward()
            # scheduler.step(float(loss_t))
            total_q_max += q_maxs.mean()
            optimizer.step()

            # gradient clipping
            # for param in net.parameters():
            #     param.grad.data.clamp_(-GRADIENT_CLIP, GRADIENT_CLIP)

            # 타겟 네트워크 갱신
            if train_cnt % SYNC_TARGET_FREQ == 0:
                log("sync target network - speed {} train / sec".format(fps))
                log(net.state_dict()['conv.0.weight'][0][0])
                tgt_net.load_state_dict(net.state_dict())

                # 보드 게시
                for name, param in net.named_parameters():
                    writer.add_histogram("learner/" + name,
                                         param.clone().cpu().data.numpy(),
                                         train_cnt)
                writer.add_scalar("learner/loss", float(loss_t), train_cnt)
                # writer.add_scalar("learner/fps", fps, train_cnt)
                writer.add_scalar("learner/Qmax",
                                  float(total_q_max / train_cnt), train_cnt)
                writer.add_scalar("buffer/replay", binfo.replay, train_cnt)
                for ano, ainfo in ainfos.items():
                    # writer.add_scalar("actor-{}/fps".format(ano),
                    #                   ainfo.speed, ainfo.frame)
                    writer.add_scalar("actor/{}-reward".format(ano),
                                      ainfo.reward, ainfo.frame)

        # 모델 발행
        publish_model(net, tgt_net, act_sock)

        if p_time is not None:
            elapsed = time.time() - p_time
            fps = 1.0 / elapsed
            log("train elapsed {:.2f} fps {:.2f}".format(elapsed, fps))

        p_time = time.time()

    writer.close()


if __name__ == '__main__':
    main()
    # Give 0MQ time to deliver
    time.sleep(1)
