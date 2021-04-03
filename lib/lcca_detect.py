from matplotlib import pyplot as plt
import scipy.stats as ss
import numpy as np
from myplot import *
from generate_data import generate_data_ramp


def Linear_CCA(input_free, output_free, input_fault, output_fault):

    fault_index = 1000

    num_sample = input_free.shape[1]
    num_samplef = input_fault.shape[1]
    dim_input = input_free.shape[0]
    dim_output = output_free.shape[0]

    mean_in = np.mean(input_free, axis=1)
    mean_out = np.mean(output_free, axis=1)
    norm_input = (input_free.T - np.tile(mean_in, (num_sample, 1))).T
    norm_output = (output_free.T - np.tile(mean_out, (num_sample, 1))).T


    norm_inputf = (input_fault.T - np.tile(mean_in, (num_samplef, 1))).T
    norm_outputf = (output_fault.T - np.tile(mean_out, (num_samplef, 1))).T

    SigmaHat12 = np.dot(norm_input, norm_output.T)     # 6*3
    SigmaHat11 = np.dot(norm_input, norm_input.T)
    SigmaHat22 = np.dot(norm_output, norm_output.T)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)  # 6*6
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)  # 3*3

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
    [U, S_2, V_] = np.linalg.svd(Tval)
    V = V_.T
    S = np.zeros([dim_input, dim_output])   # 3 * 4
    for i in range(min(dim_input, dim_output)):
        S[i, i] = S_2[i]
    rank_S = S_2.shape[0]


    P = np.dot(SigmaHat11RootInv, U[:, :rank_S])
    P_res = np.dot(SigmaHat11RootInv, U[:, rank_S:])
    L = np.dot(SigmaHat22RootInv, V[:, :rank_S])
    L_res = np.dot(SigmaHat22RootInv, V[:, rank_S:])
    P_2 = np.concatenate((P, P_res), 1).T
    L_2 = np.concatenate((L, L_res), 1).T


    Inv_input = np.linalg.inv((np.identity(dim_input) - np.dot(S, S.T))/(num_samplef-1))
    T2_in = []
    res1 = []
    Q2_1 = []
    for i in range(num_sample):
        te1 = np.dot(P_2, norm_inputf[:, i]) - np.dot(np.dot(S, L_2), norm_outputf[:, i])
        te2 = np.dot(np.dot(te1.T, Inv_input), te1)
        T2_in.append(te2)
        res1.append(te1)
        q2_1 = np.dot(te1.T, te1)
        Q2_1.append(q2_1)
    T2_in = np.array(T2_in)
    res1 = np.array(res1)
    Q2_1 = np.array(Q2_1)


    Inv_output = np.linalg.inv((np.identity(dim_output) - np.dot(S.T, S))/(num_samplef-1))
    T2_in2 = []
    res2 = []
    Q2_2 = []
    for i in range(num_sample):
        te1 = np.dot(L_2, norm_outputf[:, i]) - np.dot(np.dot(S.T, P_2), norm_inputf[:, i])
        te2 = np.dot(np.dot(te1.T, Inv_output), te1)
        T2_in2.append(te2)
        res2.append(te1)
        q2_2 = np.dot(te1.T, te1)
        Q2_2.append(q2_2)
    T2_in2 = np.array(T2_in2)
    res2 = np.array(res2)
    Q2_2 = np.array(Q2_2)

    alpha = 0.05
    thre1_T = ss.chi2.isf(alpha, Inv_input.shape[1])
    thre2_T = ss.chi2.isf(alpha, Inv_output.shape[1])
    
    rate_false1, rate_mdr1 = far_mdr_compute(T2_in, thre1_T, fault_index)
    rate_false2, rate_mdr2 = far_mdr_compute(T2_in2, thre2_T, fault_index)
    fdd = fdd_compute(T2_in2, thre2_T, fault_index)
    
    t = np.linspace(0, num_sample, num_sample)
    smin = np.min(T2_in2)
    smax = np.max(T2_in2)
    length = len(T2_in2)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 40
    plt.figure(figsize=(16,8))
    plt.yscale('log')
    plt.plot(np.arange(length), T2_in2, linewidth=4, alpha=0.7, zorder=0)
    plt.vlines(x=fault_index, ymin=smin, ymax=smax, colors='r', label='fault injection', linestyles='-.', linewidth=8, zorder=2)
    plt.hlines(xmin=0, xmax=length, y=thre2_T, colors='orange', label='detection threshold', linestyles='--', linewidth=8, zorder=3)
    print(f'T2_2, thred:{thre2_T:.3f}, FAR:{rate_false2:.2%}, MDR:{rate_mdr2:.2%}, FDD:{fdd}')
    plt.ylim([smin, smax])
    plt.legend(loc=4)
    plt.xlabel('Sample')
    plt.ylabel('Test statistic')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    saveRes('LinearCCA', T2_in2, thre2_T, fault_index)
        
    plt.show()




if __name__ == "__main__":
    input_free, output_free, input_fault, output_fault = generate_data_ramp(2)
    input_free = input_free.T
    output_free = output_free.T
    input_fault = input_fault.T
    output_fault = output_fault.T
    Linear_CCA(input_free.T, output_free.T, input_fault.T, output_fault.T)

