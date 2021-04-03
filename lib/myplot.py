import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.io import savemat


def generate_batch(input_free, output_free, input_fault, output_fault, batch_len):
    series_len = input_free.shape[0]
    input_free_batch = []
    output_free_batch = []
    input_fault_batch = []
    output_fault_batch = []
    for i in range(series_len - batch_len + 1):
        input_free_batch.append(input_free[i: (i + batch_len)])
        output_free_batch.append(output_free[i: (i + batch_len)])
        input_fault_batch.append(input_fault[i: (i + batch_len)])
        output_fault_batch.append(output_fault[i: (i + batch_len)])
    return np.array(input_free_batch), np.array(output_free_batch), np.array(input_fault_batch), np.array(output_fault_batch)


def far_mdr_compute(sta, thre, index_thre):
    rate_false = (sta[:index_thre] > thre).tolist().count(True) / index_thre
    rate_mdr = (sta[index_thre:] < thre).tolist().count(True) / (sta.shape[0] - index_thre)
    return rate_false, rate_mdr



def saveRes(name, statistics, threshold, fault_index):
    if not os.path.exists('./result/fig/'):
        os.makedirs('./result/fig/')
    savemat('./result/'+name+'.mat',
            {'statistics':statistics,
             'threshold':threshold,
             'fault_index':fault_index,
            })
    plt.savefig('./result/fig/'+name+'.png', dpi=300)


def fdd_compute(data, threshold, fault_index):
    L = len(data)
    fd_index = L
    for i in range(L-1,fault_index,-1):
        if(data[i] > threshold):
            fd_index = i-fault_index
    return fd_index

def threshold_compute(statistics):
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import MinMaxScaler

    staMin = np.min(statistics.ravel())
    staMax = np.max(statistics.ravel())
    staBand = (staMax - staMin) * 0.5
    ts = np.linspace(staMin-staBand, staMax+staBand, 1000).reshape(-1,1)
    kde = np.exp(KernelDensity().fit(statistics.reshape(-1,1)).score_samples(ts))
    distribution = MinMaxScaler(feature_range=(0,1)).fit_transform(np.cumsum(kde/1000).reshape(-1,1))

    return ts[1000-len(distribution[distribution>0.95])]


def plot_ts(test_corr_up, S_up, hidden_view1_up, hidden_view2_up, test_corr_dw, S_dw,
             hidden_view1_dw, hidden_view2_dw, outdim_size, pt_num, num_sample, saveName='None'):
    print(f'DCCA for free data: {-test_corr_up}, DCCA for fault data: {-test_corr_dw}')

    # up_1
    S2_up = np.diag(S_up)
    res1_up = []
    T2_up = []
    Q2_up = []
    Inv_output_up = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_up, S2_up.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_up = hidden_view1_up[i] - np.dot(S2_up, hidden_view2_up[i])
        res1_up.append(te1_up)
        # compute the T2
        te2_up = np.dot(np.dot(te1_up.T, Inv_output_up), te1_up)
        T2_up.append(te2_up)
        # compute the Q2
        q2_up = np.dot(te1_up.T, te1_up)
        Q2_up.append(q2_up)
    res1_up = np.array(res1_up)
    T2_up = np.array(T2_up)
    Q2_up = np.array(Q2_up)
    threshold1_T = get_thred(T2_up)
    # threshold1_Q = get_thred(Q2_up)

    # up_2
    S2_up = np.diag(S_up)
    res2_up = []
    T2_2_up = []
    Q2_2_up = []
    Inv_output_up = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_up, S2_up.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_up = hidden_view2_up[i] - np.dot(S2_up, hidden_view1_up[i])
        res2_up.append(te1_up)
        # compute the T2
        te2_up = np.dot(np.dot(te1_up.T, Inv_output_up), te1_up)
        T2_2_up.append(te2_up)
        # compute the Q2
        q2_up = np.dot(te1_up.T, te1_up)
        Q2_2_up.append(q2_up)
    res2_up = np.array(res2_up)
    T2_2_up = np.array(T2_2_up)
    Q2_2_up = np.array(Q2_2_up)
    # threshold2_T = get_thred(T2_2_up)
    # threshold2_Q = get_thred(Q2_2_up)

    # dw_1
    S2_dw = np.diag(S_dw)
    res1_dw = []
    T2_dw = []
    Q2_dw = []
    Inv_output_dw = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_dw, S2_dw.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_dw = hidden_view1_dw[i] - np.dot(S2_dw, hidden_view2_dw[i])
        res1_dw.append(te1_dw)
        # compute the T2
        te2_dw = np.dot(np.dot(te1_dw.T, Inv_output_dw), te1_dw)
        T2_dw.append(te2_dw)
        # compute the Q2
        q2_dw = np.dot(te1_dw.T, te1_dw)
        Q2_dw.append(q2_dw)
    res1_dw = np.array(res1_dw)
    T2_dw = np.array(T2_dw)
    Q2_dw = np.array(Q2_dw)

    # dw_2
    S2_dw = np.diag(S_dw)
    res2_dw = []
    T2_2_dw = []
    Q2_2_dw = []
    Inv_output_dw = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_dw, S2_dw.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_dw = hidden_view2_dw[i] - np.dot(S2_dw, hidden_view1_dw[i])
        res2_dw.append(te1_dw)
        # compute the T2
        te2_dw = np.dot(np.dot(te1_dw.T, Inv_output_dw), te1_dw)
        T2_2_dw.append(te2_dw)
        # compute the Q2
        q2_dw = np.dot(te1_dw.T, te1_dw)
        Q2_2_dw.append(q2_dw)
    res2_dw = np.array(res2_dw)
    T2_2_dw = np.array(T2_2_dw)
    Q2_2_dw = np.array(Q2_2_dw)


    fault_index = num_sample - 1000 + 1
    threshold2_T = get_thred(T2_2_dw[0:fault_index])
#     print('fault index =', fault_index)
#     print('threshold =', threshold2_T)


    rate_false_T1, rate_mdr_T1 = far_mdr_compute(T2_dw, threshold1_T, fault_index)
    rate_false_T2, rate_mdr_T2 = far_mdr_compute(T2_2_dw, threshold2_T, fault_index)

    t = np.linspace(0, pt_num, pt_num)
    fdd = fdd_compute(T2_2_dw, threshold2_T, fault_index)
    smin = np.min(T2_2_dw)
    smax = np.max(T2_2_dw)
    length = len(T2_2_dw)
    

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 40
    plt.figure(figsize=(16,8))
    plt.yscale('log')
    plt.plot(np.arange(length), T2_2_dw[:], linewidth=4, alpha=0.7, zorder=0)
    plt.vlines(x=fault_index, ymin=smin, ymax=smax, colors='r', label='fault injection', linestyles='-.', linewidth=8, zorder=2)
    plt.hlines(xmin=0, xmax=length, y=threshold2_T, colors='orange', label='detection threshold', linestyles='--', linewidth=8, zorder=3)
    print(f'T2_2, thred:{threshold2_T:.3f}, FAR:{rate_false_T2:.2%}, MDR:{rate_mdr_T2:.2%}, FDD:{fdd}')
    plt.ylim([smin, smax])
    plt.legend(loc=4)
    plt.xlabel('Sample')
    plt.ylabel('Test statistic')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if not (saveName == 'None'):
        saveRes(saveName, T2_2_dw, threshold2_T, fault_index)
        
    plt.show()



def get_thred(T2_statistic, alpha=0.95):
    # T2_statistic (1, None)
    data = T2_statistic.reshape(-1, 1)
    Min = np.min(data)
    Max = np.max(data)
    Range = Max - Min
    x_start = Min - Range
    x_end = Max + Range
    nums = 2 ** 12
    dx = (x_end-x_start)/(nums-1)
    data_plot = np.linspace(x_start, x_end, nums)[:, np.newaxis]

    # choose the best bandwidth
    data_median = np.median(data)
    new_median = np.median(np.abs(data - data_median)) / 0.6745
    bw = new_median * ((4 / (3 * data.shape[0])) ** 0.2)

    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
    log_dens = kde.score_samples(data_plot)
    pdf = np.exp(log_dens).reshape(-1, 1)

    CDF = 0
    index = 0
    while CDF <= alpha:
        CDF += pdf[index]*dx
        index += 1
    return np.squeeze(data_plot[index])