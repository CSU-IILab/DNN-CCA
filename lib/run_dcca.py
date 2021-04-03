import os
import tensorflow as tf
import time
import random
from keras import backend as K
from myplot import *
from generate_data import generate_data_ramp
import shutil
from scipy.io import savemat

import argparse
parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--model', type=str, default = 'cnn')
parser.add_argument('--train', type=str, default = 'False')
parser.add_argument('--epoch', type=int, default = 100)
parser.add_argument('--window-len', type=int, default = 70)
parser.add_argument('--output-dim', type=int, default = 30)
parser.add_argument('--model-index', type=int, default = 1)
args = parser.parse_args()
model = args.model
retrain = True if args.train == 'True' else False
n_epochs = args.epoch
batch_len = args.window_len
outdim_size = args.output_dim
model_index = args.model_index

if model == 'cnn':
    from model_cnn import DeepCCA_cnn as DeepCCA_model
elif model == 'lstm':
    from model_lstm import DeepCCA_lstm as DeepCCA_model
elif model == 'gru':
    from model_gru import DeepCCA_gru as DeepCCA_model
elif model == 'attention':
    from model_attention import DeepCCA_attention as DeepCCA_model


saveName = model
model_save_rootpath = './model'
dir_path = os.path.join(model_save_rootpath, f'{model}_{batch_len}_{outdim_size}_{model_index}')
if (retrain == True):
    try:
        shutil.rmtree(dir_path)
    except:
        pass
model_save_path = os.path.join(dir_path, "best.ckpt")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

log_save_rootpath = './model/'
dir_path = os.path.join(log_save_rootpath, f'log_{model}_{batch_len}_{outdim_size}_{model_index}')
if (retrain == True):
    try:
        shutil.rmtree(dir_path)
    except:
        pass
log_save_path = dir_path
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

    
learning_rate = 0.01
momentum = 0.5
batch_size = 256
num_sample = 2000 - batch_len + 1
input_size1 = 3
input_size2 = 2
input_2d = batch_len
reg_par = 2
use_all_singular_values = True

input_free, output_free, input_fault, output_fault = generate_data_ramp(2)
input_free = input_free.T               
output_free = output_free.T
input_fault = input_fault.T
output_fault = output_fault.T


num_sam = input_free.shape[0]
num_samf = input_fault.shape[0]
mean_in = np.mean(input_free, axis=0)
mean_out = np.mean(output_free, axis=0)
input_free = input_free - np.tile(mean_in, (num_sam, 1))
output_free = output_free - np.tile(mean_out, (num_sam, 1))
input_fault = input_fault - np.tile(mean_in, (num_samf, 1))
output_fault = output_fault - np.tile(mean_out, (num_samf, 1))


input_free_batch, output_free_batch, input_fault_batch, output_fault_batch\
    = generate_batch(input_free, output_free, input_fault, output_fault, batch_len)

tf.reset_default_graph()
rand_num = random.randint(0, 1000)
K.set_learning_phase(1)
dcca_model = DeepCCA_model(input_size1, input_size2, outdim_size, reg_par, input_2d)

input_view1 = dcca_model.input_view1
input_view2 = dcca_model.input_view2
hidden_view1 = dcca_model.output_view1
hidden_view2 = dcca_model.output_view2
neg_corr = dcca_model.neg_corr
S = dcca_model.S
rate_ = dcca_model.rate


tf.summary.scalar('correlation', -neg_corr)
merged = tf.summary.merge_all()

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
train_op = tf.train.AdamOptimizer().minimize(neg_corr,
                                                                        var_list=tf.trainable_variables())
iterations = 0


saver = tf.train.Saver()
loss = 0
last_loss = 0
loss_his = []
corr1 = []
corr2 = []
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as sess:

    if not os.path.exists(model_save_path + ".index"):
        print("Learning begins...")
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_save_path, sess.graph)


        start_time = time.time()
        for epoch in range(n_epochs):
            for batch in range(int(input_free_batch.shape[0]/batch_size)):

                X1_batch = input_free_batch[(batch * batch_size): ((batch + 1) * batch_size)]
                X2_batch = output_free_batch[(batch * batch_size): ((batch + 1) * batch_size)]

                _, neg_corr_val, summary = sess.run([train_op, neg_corr, merged],
                                           feed_dict={input_view1: X1_batch, input_view2: X2_batch
                                           })
                train_writer.add_summary(summary, batch + epoch * batch_size)

                loss = neg_corr_val
                loss_his.append(loss)
                if last_loss > loss:
                    last_loss = loss
                    if not os.path.exists(model_save_path):
                        os.mkdir(model_save_path)
                        print("create the directory: %s" % model_save_path)
                    saver.save(sess, model_save_path)


                if iterations % 100 == 0:
                    time_node = time.time()
                    print("epoch: " + str(epoch) + " correlation for train:", -neg_corr_val/outdim_size, " time costs " + str(time_node - start_time) + "s")

                    test_corr, S_, hidden_view1_, hidden_view2_ = sess.run([neg_corr, S, hidden_view1, hidden_view2], feed_dict=
                                                        {input_view1: input_free_batch, input_view2: output_free_batch})
                    print("correlation of normal data: ", -test_corr/outdim_size)
                    corr1.append(-test_corr)

                    test_corr3, S_3, hidden_view1_3, hidden_view2_3 = sess.run([neg_corr, S, hidden_view1, hidden_view2], feed_dict=
                                                        {input_view1: input_fault_batch, input_view2: output_fault_batch})
                    print("correlation of faulty data: ", -test_corr3/outdim_size)
                    corr2.append(-test_corr3)
                    print("\n")

                iterations += 1
#         savemat('corr.mat', {'corr1':corr1, 'corr2':corr2})



    else:
        K.set_learning_phase(0)
        pt_num = 2000 - batch_len + 1
        saver.restore(sess, model_save_path)
        test_corr_up, S_up, hidden_view1_up, hidden_view2_up = sess.run([neg_corr, S, hidden_view1, hidden_view2], feed_dict=
                                                       {input_view1: input_free_batch, input_view2: output_free_batch, rate_: 1})
        test_corr_dw, S_dw, hidden_view1_dw, hidden_view2_dw = sess.run([neg_corr, S, hidden_view1, hidden_view2], feed_dict=
                                                       {input_view1: input_fault_batch, input_view2: output_fault_batch, rate_: 1})
        plot_ts(test_corr_up, S_up, hidden_view1_up, hidden_view2_up, test_corr_dw, S_dw,
                                                          hidden_view1_dw, hidden_view2_dw, outdim_size, pt_num, num_sample, saveName)
