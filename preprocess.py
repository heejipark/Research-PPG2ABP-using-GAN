#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pyvital.arr as arr
from scipy import signal as sig
import numpy as np

def process_beat(seg):
    """
    check if arterial pressure meets requirements and return average beat
    :param seg: numpy-array type segment
    :return: mean std of beats and avgbeat
    """
    # return: mean_std, avg_beat
    minlist, maxlist = arr.detect_peaks(seg, 100)

    # beat lengths
    beatlens = []
    beats = []
    beats_128 = []
    for i in range(1, len(maxlist) - 1):
        beatlen = maxlist[i] - maxlist[i - 1]  # in samps
        pp = seg[maxlist[i]] - seg[minlist[i - 1]]  # pulse pressure

        # hr 20 - 200 만 허용
        if pp < 20:
            return 0, []
        elif beatlen < 30:  # print('{} too fast rhythm {}'.format(id, beatlen))
            return 0, []
        elif beatlen > 300 or (i == 1 and maxlist[0] > 300) or (i == len(maxlist) - 1 and len(seg) - maxlist[i] > 300):
            # print ('{} too slow rhythm {}', format(id, beatlen))
            return 0, []
        else:
            beatlens.append(beatlen)
            beat = seg[minlist[i - 1]: minlist[i]]
            beats.append(beat)
            resampled = sig.resample(beat, 128)
            beats_128.append(resampled)

    if not beats_128:
        return 0, []

    avgbeat = np.array(beats_128).mean(axis=0)

    nucase_mbeats = len(beats)
    if nucase_mbeats < 10:  # print('{} too small # of rhythm {}'.format(id, nucase_mbeats))
        return 0, []
    else:
        meanlen = np.mean(beatlens)
        stdlen = np.std(beatlens)
        if stdlen > meanlen * 0.2:  # print('{} irregular thythm', format(id))
            return 0, []

    # beat 내부의 correlation이 0.9이하인 것들의 표준편차를 구함
    beatstds = []
    for i in range(len(beats_128)):
        if np.corrcoef(avgbeat, beats_128[i])[0, 1] > 0.9:
            beatstds.append(np.std(beats[i]))

    if len(beatstds) * 2 < len(beats):
        return 0, []

    return np.mean(beatstds), avgbeat





