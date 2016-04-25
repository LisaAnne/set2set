import sys
sys.path.insert(0, '../../python/')
import caffe
import copy
import pdb
from utils.python_utils import *

process_prototxt = 'prototxts/deploy_process_set2seq_train.prototxt'
write_prototxt = 'prototxts/deploy_write_set2seq_train.prototxt'
model_weights = 'snapshots/set2seq_iter_10000.caffemodel'
max_batch_size = 100
T = 5
max_value = 1000
num_process_steps = 1
#test_set = 'utils/data/ls_%d_mv_%d_test.txt' %(T, max_value)
test_set = 'utils/data/ls_%d_int_test.txt' %(T)
max_value=1
process_net = caffe.Net(process_prototxt, model_weights, caffe.TEST)
write_net = caffe.Net(write_prototxt, model_weights, caffe.TEST)

data = read_data(test_set)
len_data = data.shape[0]
output = np.zeros((len_data, T))

for i in range(0, len_data, max_batch_size):
  batch_size = min(max_batch_size, len_data-i)

  #un process block
  process_net.blobs['rand_data'].reshape(batch_size, T, max_value)
  if max_value > 1:
    process_net.blobs['rand_data'].data[...] = mat_to_one_hot(data[i:i+batch_size,...], max_value)
  else:
    process_net.blobs['rand_data'].data[:,:,0] = data[i:i+batch_size,...]
  #pdb.set_trace()
  #initialize size for internal LSTM layers
  process_net.blobs['q_init'].reshape(1, batch_size, T)
  process_net.blobs['c_0'].reshape(1, batch_size, T)
  process_net.blobs['x'].reshape(1, batch_size, T*2)
  process_net.blobs['cont_0'].reshape(1, batch_size)
  if 'cont_all' in process_net.blobs.keys():
    process_net.blobs['cont_0'].reshape(1, batch_size)
  process_net.forward()

  #run write block
  write_net.blobs['LSTM_input_0'].reshape(batch_size, 1, max_value)
  write_net.blobs['message'].reshape(batch_size, T, T)
  write_net.blobs['q_star_T'].reshape(1, batch_size, T*2) #q_star_T is the cell statefrom previous block
  write_net.blobs['gen_c_0'].reshape(1, batch_size, T*2)
  write_net.blobs['cont_0'].reshape(1, batch_size)

  LSTM_input_0 = np.zeros((batch_size, 1, max_value))
  message = copy.deepcopy(process_net.blobs['read_ip_0'].data) 
  q_star_T = copy.deepcopy(process_net.blobs['q_star_%d' %num_process_steps].data) 
  gen_c_0 = np.zeros((1, batch_size, T*2))
  cont_0 = np.zeros((1, batch_size))

  for t in range(T):
    write_net.blobs['LSTM_input_0'].data[...] = LSTM_input_0
    write_net.blobs['message'].data[...] = message
    write_net.blobs['q_star_T'].data[...] = q_star_T
    write_net.blobs['gen_c_0'].data[...] = gen_c_0
    write_net.blobs['cont_0'].data[...] = cont_0 
    write_net.forward()
    #put togehter next LSTM input
    input_index = np.argmax(write_net.blobs['probs'].data, axis=1).squeeze()
    LSTM_input_0 = copy.deepcopy(process_net.blobs['rand_data'].data[range(batch_size),input_index,:])  
    LSTM_input_0 = LSTM_input_0.reshape((batch_size, 1, max_value))
    q_star_T = copy.deepcopy(write_net.blobs['gen_h_1'].data[...])
    gen_c_0 = copy.deepcopy(write_net.blobs['gen_c_1'].data[...])
    cont_0 = np.ones((1, batch_size))
    if max_value > 1:
      output[i:i+batch_size,t] = np.where(LSTM_input_0[:,0,:] == 1)[1]
    else:
      output[i:i+batch_size,t] = LSTM_input_0.squeeze()


incorrect = np.sum(np.sum(np.sort(data, axis=1) - output, axis=1) > 0)
print "Error: %f" %(float(incorrect)/output.shape[0])












