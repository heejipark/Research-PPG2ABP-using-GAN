# Required libraries
import os
import random
import vitaldb
import pyvital
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import preprocess as pre
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument('--PPG', type=str, default='SNUADC/PLETH', help='track name for PPG')
parser.add_argument('--ABP', type=str, default='SNUADC/ART', help='track name for ABP')
parser.add_argument('--SRATE', type=int, default=100, help='srate')
parser.add_argument('--MAX_CASE', type=int, default=1000, help='Maximum number of patients')
parser.add_argument('--SEC', type=int, default=25, help='seconds')
parser.add_argument('--case_sample', type=int, default=0, help='Number of valid case')
parser.add_argument('--cachefile', type=str, default='datasets.npz', help='File name for store')
opt = parser.parse_args()

valid_mask = []
ppg_sets = []
abp_sets = []
for cid in tqdm(range(1, opt.MAX_CASE+1)):
    vals = vitaldb.load_case(caseid=cid, tnames=[opt.PPG, opt.ABP], interval=1/opt.SRATE)
    for i in range(0, len(vals), opt.SEC * opt.SRATE):
        segx = vals[i : i + opt.SEC*opt.SRATE]
        ppg, abp = segx[:,0], segx[:,1]
        
        if len(segx) < opt.SEC * opt.SRATE: # 20초 이하의 데이터는 사용 x
            continue
        ###########################################
        # Check the validity of the segemnt
        # Valid condition
        # (1) 0 <= PPG <= 100
        # (2) 20 <= ABP <= 200
        # (3) mstd_val(abp) > 0
        # Else, remove
        ###########################################
        
        valid = True
        mstd_val, _ = pre.process_beat(abp)
        
        if np.isnan(ppg).any() or np.isnan(abp).any(): # delete non-value
            valid = False
        if (ppg < 0).any() or (ppg > 100).any():
            valid = False
        if (abp < 20).any() or (abp > 250).any():
            valid = False
        if np.array(mstd_val) <= 0:
            valid = False

        if valid:
            # Check ppg peak point
            ppg = pyvital.arr.exclude_undefined(ppg)
            ppg_peaks = pyvital.arr.detect_peaks(ppg, 100) 
            
            # Check abp peak point
            abp = pyvital.arr.exclude_undefined(abp)
            abp_peaks = pyvital.arr.detect_peaks(abp, 100)
                        
            # Mapping the first peak of PPG and the second peak of ABP
            try:
                pid, aid = ppg_peaks[0][0], abp_peaks[0][1]
                new_ppg = ppg[pid:pid+2000]
                new_abp = abp[aid:aid+2000]
                
                if len(new_ppg) != 2000 or len(new_abp) != 2000:
                    continue
                ppg_sets.append(new_ppg)
                abp_sets.append(new_abp)                
                opt.case_sample += 1
            except:
                continue

#### After Preprocessing ------------------------------------
ppg_sets = savgol_filter(ppg_sets, 31, 3)
abp_sets = savgol_filter(abp_sets, 31, 3)

np.savez(opt.cachefile, ppg_sets=ppg_sets, abp_sets=abp_sets) # Save cahce file