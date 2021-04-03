import numpy as np


def generate_data_ramp(fig):
    t1 = 1.99 * np.random.random((2000, 1)) + 0.01
    t2 = 1.99 * np.random.random((2000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (2000, 1))
    e2 = np.random.normal(0, 0.1, (2000, 1))
    e3 = np.random.normal(0, 0.1, (2000, 1))
    e4 = np.random.normal(0, 0.1, (2000, 1))
    e5 = np.random.normal(0, 0.1, (2000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    output2f[0: 1000, :] = output2[0: 1000, :]
    output2f[1000: 2000, :] = output2[1000: 2000, :] + fault 

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)
    return input_free.T, output_free.T, input_fault.T, output_fault.T    # (3 * 2000)



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




