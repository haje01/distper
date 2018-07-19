"""리플레이 버퍼 모듈."""
import pickle

import zmq
import numpy as np
from tensorboardX import SummaryWriter

from common import ExperienceBuffer, async_recv, ActorInfo, ENV_NAME

REPLAY_SIZE = 500000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32


def average_actor_info(infos):
    """액터 정보를 평균."""
    infos = ActorInfo(*zip(*infos))
    tmp = ActorInfo(*np.mean(infos, axis=1))
    info = ActorInfo(tmp.episode, int(tmp.frame), tmp.reward, tmp.speed)
    return info


buffer = ExperienceBuffer(REPLAY_SIZE)
context = zmq.Context()

writer = SummaryWriter(comment="-" + ENV_NAME)

# 액터/러너에게서 받을 소켓
recv = context.socket(zmq.PULL)
recv.bind("tcp://*:5558")

# 러너에게 보낼 소켓
learner = context.socket(zmq.REP)
learner.bind("tcp://*:5555")

actor_infos = []  # 액터들이 보낸 정보

# 반복
while True:
    # 액터에게서 리플레이 정보 받음
    payload = async_recv(recv)
    if payload is not None:
        ebuf, info = pickle.loads(payload)
        actor_infos.append(info)
        # 리플레이 버퍼에 병합
        buffer.merge(ebuf)
        print("receive replay - buffer size {}".format(len(buffer)))

    # 러너가 배치를 요청했으면 보냄
    if async_recv(learner) is not None:
        if len(actor_infos) > 0:
            info = average_actor_info(actor_infos)
        else:
            info = None

        if len(buffer) < REPLAY_START_SIZE:
            payload = b'not enough'
            print("not enough data - buffer size {}".format(len(buffer)))
        else:
            # 충분하면 샘플링 후 보냄
            batch = buffer.sample(BATCH_SIZE)
            payload = pickle.dumps((batch, info))
            print("send batch")

        if info is not None:
            writer.add_scalar("buffer/replay", len(buffer), info.frame)

        # 전송
        learner.send(payload)
