import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from pythonosc import udp_client

from itertools import cycle
import numpy as np

from allopy.chronos import rhythm_trees as r_trees
from allopy.chronos.rhythm_trees import rt_algorithms as rt_alg
from allopy.chronos import temporal_units as ut
from allopy.tonos.JI import combination_sets as cps
from allopy.tonos import fold_freq
from allopy.topos.sequences import Norg

client = udp_client.SimpleUDPClient('127.0.0.1', 57120)

rts = [
    r_trees.RT(time_signature='4/4', subdivisions=(3,(1,(2,1)),2,1,(1,(1,1,1)))),
    r_trees.RT(time_signature='15/16', subdivisions=(3,2,1,3,1,2,3)),
    r_trees.RT(time_signature='4/3', subdivisions=((4,(3,(8,(3,4)))),-3)),
    r_trees.RT(time_signature='7/5', subdivisions=((4,(1,1,1)),(3,(1,)*8),-5)),
    r_trees.RT(time_signature='21/12', subdivisions=(11,7,5,3)),
]

# u_t = ut.UT(tempus='7/5', prolatio=((4,(1,1,1)),(3,(1,)*8),-5), tempo=60, beat='1/5')
i = -1
u_t = ut.UT.from_tree(rts[i], tempo=60, beat='1/4')
u_t.tempo = 36
u_t.beat = f'1/{u_t.tempus.denominator}'
hx = cps.Hexany()
# freqs = cycle([fold_freq(1.0 * 333.0 * np.random.choice([r, 1/r])) for r in hx.ratios])
# amps = cycle([np.random.uniform(0.05, 0.25) for _ in range(5)])

print(u_t.tree)
events = [
    # ('syn', 0.2, 'dur', 1, 'freq', 500, 'amp', 0.7),
    # ('saw', 0.5, 'dur', 1.5, 'freq', 440, 'amp', 0.6),
    # ('bassDrum', 0.0, 'dur', 0.4)
]

rt = r_trees.RT(duration=11, time_signature='9/4', subdivisions=rt_alg.autoref((7,5,13,3,11)))
rt_mel = r_trees.RT(duration=11, time_signature='9/4', subdivisions=rt_alg.autoref(rt_alg.rhythm_pair((1,3,5))))

for i, n in enumerate(hx.factors):
    freqs = cycle([fold_freq(n * 166.5 * np.random.choice([r, 1/r]), lower=83.25) for r in hx.ratios])
    amps = cycle([np.random.uniform(0.005, 0.05) for _ in range(np.random.randint(3, 7))])
    for start, duration in ut.UT.from_tree(rt.rotate(i), tempo=36, beat='1/16'):
    # for start, duration in u_t:
        if duration < 0: continue
        # print(start, duration)
        freq = next(freqs)
        amp = next(amps)
        events.append(('syn', start, 'dur', duration*3.33, 'freq', freq, 'amp', amp))

# m_freq = cycle([fold_freq(666.0 * hx.ratios[Norg.inf_num(i) % len(hx.ratios)], lower=166.5, upper=999.0) for i in range(11)])
# for i, (start, duration) in enumerate(ut.UT.from_tree(rt_mel, tempo=36, beat='1/16')):
#     if duration < 0 or i < 5: continue
#     freq = next(m_freq)
#     events.append(('theremin', start, 'dur', duration, 'freq', freq, 'amp', 0.03))

for event in events:
    structured_event = [event[0], event[1]] + list(event[2:])
    client.send_message("/storeEvent", structured_event)

print("Events have been sent.")