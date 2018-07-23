"""리플레이 버퍼 모듈."""
import pickle
from collections import defaultdict

import zmq
import numpy as np

from common import ExperiencePriorityBuffer, async_recv, ActorInfo,\
    BufferInfo, get_logger

BUFFER_SIZE = 100000  # 원래는 2,000,000
START_SIZE = 10000    # 원래는 50,000
BATCH_SIZE = 256   # 전송할 배치 크기


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

memory = ExperiencePriorityBuffer(BUFFER_SIZE)
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
        actor_id, batch, prios, ainfo = pickle.loads(payload)
        actor_infos[actor_id].append(ainfo)
        # 리플레이 버퍼에 병합
        for prio, sample in zip(prios, batch):
            memory._append(prio, sample)
        log("receive replay - memory size {}".format(len(memory)))

    # 러너가 배치를 요청했으면 보냄
    payload = async_recv(learner)
    if payload is not None:
        # 러너 학습 에러 버퍼에 반영
        idxs, errors = pickle.loads(payload)
        if idxs is not None:
            print("update by learner")
            print("  idxs: {}".format(idxs))
            print("  error: {}".format(errors))
            for i in range(len(errors)):
                memory.update(idxs[i], errors[i])

        # 러너가 보낸 베치와 에러
        if len(actor_infos) > 0:
            ainfos = average_actor_info(actor_infos)
        else:
            ainfos = None

        if len(memory) < START_SIZE:
            payload = b'not enough'
            log("not enough data - memory size {}".format(len(memory)))
        else:
            # 충분하면 샘플링 후 보냄
            batch, idxs, prios = memory.sample(BATCH_SIZE)
            binfo = BufferInfo(len(memory))
            payload = pickle.dumps((batch, idxs, ainfos, binfo))
            log("send batch")

        # 전송
        learner.send(payload)
