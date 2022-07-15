# Draw a graph
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def graph(k):
    # Path ----------------------------------------------------------------
    datapath = 'data1.npz'
    imgpath1 = 'ppg_abp.png'
    imgpath2 = 'compared_generated_abp.png'
    imgpath3 = 'compared_generated_abp_with_filter.png'

    # Load data ----------------------------------------------------------------
    result = np.load(datapath)
    real_abp_sets = result['real_abp']
    real_ppg_sets = result['real_ppg']
    fake_abp_sets = result['fake_abp']
    fake_ppg_sets = result['fake_ppg']


    # Plot generated PPG and generated ABP --------------------------------------
    plt.figure(figsize=(20,5))
    plt.subplot(2,1,1)
    plt.plot(real_abp_sets[k], color='g', label='Ground-truth ABP')
    plt.plot(fake_abp_sets[k], color='r', label='Generated ABP')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(real_ppg_sets[k], color='g', label='Ground-truth PPG')
    plt.plot(fake_ppg_sets[k], color='r', label='Generated PPG')
    plt.legend()
    plt.savefig(imgpath2)


    # Plot comparison between generated ABP and ground-truth ABP" ----------------
    for i in range(0,10):
        plt.figure(figsize=(20,5))
        plt.plot(real_abp_sets[i], color='g', label='Ground-truth ABP')
        plt.plot(fake_abp_sets[i], color='r', label='Generated ABP')
        plt.legend()
        plt.show()
    plt.savefig(imgpath2)


    # Plot comparison between generated and filtered ABP and ground-truth ABP" ----
    fake_filtered_abp = savgol_filter(fake_abp_sets, 31, 3)
    for i in range(10):
        plt.figure(figsize=(20,5))
        plt.plot(real_abp_sets[i], color='g', label='real_ABP')
        plt.plot(fake_filtered_abp[i], color='r', label='fake_ABP')
        plt.legend()
        plt.show()
    plt.savefig(imgpath3)
