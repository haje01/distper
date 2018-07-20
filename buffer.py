"""리플레이 버퍼 모듈."""
import pickle
from collections import defaultdict

import zmq
import numpy as np

from common import ExperienceBuffer, async_recv, ActorInfo, BufferInfo,\
    get_logger

REPLAY_SIZE = 100000  # FIXME 2000000
REPLAY_START_SIZE = 10000  # FIXME 10000
BATCH_SIZE = 256


def average_actor_info(infos):
    """액터별로 정보 평균."""
    result = {}
    for ano, infos in infos.items():
        infos = ActorInfo(*zip(*infos))
        tmp = ActorInfo(*np.mean(infos, axis=1))
        info = ActorInfo(tmp.episode, int(tmp.frame), tmp.reward, tmp.speed)
        result[ano] = info
    return result

log = get_logger()

buffer = ExperienceBuffer(REPLAY_SIZE)
context = zmq.Context()

# 액터/러너에게서 받을 소켓
recv = context.socket(zmq.PULL)
recv.bind("tcp://*:5558")

# 러너에게 보낼 소켓
learner = context.socket(zmq.REP)
learner.bind("tcp://*:5555")

actor_infos = defaultdict(list)  # 액터들이 보낸 정보

# 반복
while True:
    # 액터에게서 리플레이 정보 받음
    payload = async_recv(recv)
    if payload is not None:
        actor_no, ebuf, ainfo = pickle.loads(payload)
        actor_infos[actor_no].append(ainfo)
        # 리플레이 버퍼에 병합
        buffer.merge(ebuf)
        log("receive replay - buffer size {}".format(len(buffer)))

    # 러너가 배치를 요청했으면 보냄
    if async_recv(learner) is not None:
        if len(actor_infos) > 0:
            ainfos = average_actor_info(actor_infos)
        else:
            ainfos = None

        if len(buffer) < REPLAY_START_SIZE:
            payload = b'not enough'
            log("not enough data - buffer size {}".format(len(buffer)))
        else:
            # 충분하면 샘플링 후 보냄
            batch = buffer.sample(BATCH_SIZE)
            binfo = BufferInfo(len(buffer))
            payload = pickle.dumps((batch, ainfos, binfo))
            log("send batch")

        # 전송
        learner.send(payload)
