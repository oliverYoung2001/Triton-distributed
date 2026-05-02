import json
from glob import glob
from copy import deepcopy
import itertools
from typing import Callable
import sys

# {
#     "ph": "X", "cat": "kernel", "name": "nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT", "pid": 7, "tid": 7,
#     "ts": 429685234108.312, "dur": 35174.471,
#     "args": {
#       "External id": 36,      "queued": 0, "device": 7, "context": 1,      "stream": 7, "correlation": 4265,      "registers per thread": 168,      "shared memory": 188636,      "blocks per SM": 1.000000,      "warps per SM": 12.000000,      "grid": [2, 39, 1],      "block": [384, 1, 1],      "est. achieved occupancy %": 0
#     }
#   },

def find_node_all(trace_events, cats=None, name=None, tid=None, cond=None):
    ret = []
    if name is not None and not isinstance(name, list):
        name = [name]
    for event in trace_events:
        # if hasattr(event, 'ts') and event['ts'] == 1698933768473329:
        #     print(f'{event}')
        if (not cats or ('cat' in event.keys() and event['cat'] in cats)) \
        and (not name or ('name' in event.keys() and event['name'] in name)) \
        and (not tid or ('tid' in event.keys() and event['tid'] == tid)) \
        and (not cond or cond(event)):
                ret.append(event)
    return ret

# def get_annotations(
#     events, prefix="SCH-", pid=None, rename: Callable[[str], str] = None
# ):
#     ret = []
#     for e in events:
#         if e.get("cat", None) == "user_annotation" and e["name"].startswith(prefix):
#             # SCH-RECV_FORWARD-0-True
#             v = deepcopy(e)
#             # v['cname']='black' # TODO
#             if rename is not None:
#                 v["name"] = rename(v["name"])
#             ret.append(v)
#             if pid is not None:
#                 v["pid"] = 0
#                 v["tid"] = pid
#             # print(v)
#     return ret

# def strip_name(s: str):
#     types = {
#         "RECV_FORWARD": "recv_F",
#         "RECV_BACKWARD": "recv_B",
#         "SEND_FORWARD": "send_F",
#         "SEND_BACKWARD": "send_B",
#         "F": "F",
#         "B": "B",
#         "W": "W",
#     }
#     op = s.split("-")[1]
#     return "-".join([types.get(op, op)] + s.split("-")[2:])

def recursive_modify(e, prop_dict):
    for k, v in prop_dict.items():
        if k not in e.keys():
            continue
        if isinstance(v, dict):
            recursive_modify(e[k], v)
        elif hasattr(v, "__call__"):
            e[k] = v(e[k])
        else:
            e[k] = v

def modify_annotations(events, prop_dict: dict):
    for e in events:
        recursive_modify(e, prop_dict)
    return events

# def remove_device_id(events):
#     for e in events:
#         if 'args' in e.keys() and 'device' in e['args'].keys():
#             e['args'].pop('device')
#     return events

def main(ROOT_PATH, rank_start, rank_end, rank_step):
    merged_trace = None
    stream_id2name = {7: 'Comp', 27: "Send", 31: "Recv"}
    global_sync_point = None
    for rank in range(rank_start, rank_end, rank_step):
        # fns = glob(path_prefix + f"{rank}.json")
        fns = glob(ROOT_PATH + f"/*r{rank}[!0-9]*pt.trace.json")
        print(f'[Rank{rank}] {fns}')
        assert len(fns) == 1
        fn = fns[0]
        trace = json.load(open(fn))
        events = find_node_all(trace["traceEvents"], cats=['kernel'])
        events = modify_annotations(events, {"pid": 0, "args": {'device': 0}})
        # events = modify_annotations(events, {'tid': lambda t: f'Rank{rank}|{stream_id2name[t]}'})
        events = modify_annotations(events, {'tid': lambda t: f'Rank{rank}|{t}'})
        # sync all ranks with rank0
        # ar_kernels = find_node_all(events, name=['ncclDevKernel_AllReduce_Sum_u8_TREE_LL(ncclDevComm*, unsigned long, ncclWork*)'])
        ar_kernels = find_node_all(events, cond=lambda e: 'name' in e.keys() and 'ncclDevKernel_AllReduce' in e['name'])
        # print(f'[Rank{rank}] {ar_kernels}')
        # assert len(ar_kernels) > 0
        if len(ar_kernels) > 0:
            # "ts": 446646766022.791, "dur": 24347.285,
            sync_point = ar_kernels[0]['ts'] + ar_kernels[0]['dur']
            if rank == rank_start:
                global_sync_point = sync_point
            else:
                sync_offset = global_sync_point - sync_point
                events = modify_annotations(events, {'ts': lambda t: t+sync_offset})
        else:
            print(f'[WARN] Skip synchronizing all ranks!')
        if merged_trace is None:
            merged_trace = trace
            merged_trace["traceEvents"] = []
        merged_trace["traceEvents"] += events
        print(f'Imported {len(merged_trace["traceEvents"])} events from {fn}')
    # remove "deviceProperties": [], and "distributedInfo": {},
    merged_trace['deviceProperties'] = []
    merged_trace['distributedInfo'] = {}
    json.dump(merged_trace, open(f"{ROOT_PATH}/merged.pt.trace.json", "w"))
    print(f"Generated merged trace {ROOT_PATH}/merged.pt.trace.json")


if __name__ == '__main__':
    start, end, step = 0, 16, 1
    # ROOT_PATH = './logs/tb/mamba2_attn_cp_w16_20260106_015736'
    if len(sys.argv) > 1:
        ROOT_PATH = sys.argv[1]
        # f"/data0/tsinghua/huangkz/logs/trace_ite_4-0801-073644-rank_{rank}/*.pt.trace.json"
    else:
        print(
            f"Usage: python {sys.argv[0]} <root_path> [rank_start={start}] [end={end}] [step={step}]"
        )
        print(
            f"Example: python ./logs/tb/mamba2_attn_cp_w16_20260106_015736"
        )
        exit()
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    if len(sys.argv) > 3:
        end = int(sys.argv[3])
    if len(sys.argv) > 4:
        step = int(sys.argv[4])
    print(f'{start}, {end}, {step}')
    main(ROOT_PATH, start, end, step)