import sys
sys.path.insert(0, '../../python/')
sys.path.insert(0, 'utils/')
import caffe
import copy
import pdb
from utils.python_utils import *
from caffe.proto import caffe_pb2
caffe.set_device(0)
caffe.set_mode_gpu()
import argparse
from build_set2seq import *
from build_test_data import *
import os

def test_set2seq(tag, T=10, message_size=10, num_process_steps=1):
  process_prototxt = 'prototxts/deploy_process_%s_train.prototxt' %tag
  write_prototxt = 'prototxts/deploy_write_%s_train.prototxt' %tag
  model_weights = 'snapshots/%s_iter_10000.caffemodel' %tag
  max_batch_size = 100
  max_value=1
  #test_set = 'utils/data/ls_%d_mv_%d_test.txt' %(T, max_value)
  test_set = 'utils/data/ls_%d_int_test.txt' %(T)

  if not os.path.isfile(test_set):
    build_test_data(T, 1)

  process_net = caffe.Net(process_prototxt, model_weights, caffe.TEST)
  write_net = caffe.Net(write_prototxt, model_weights, caffe.TEST)
  
  
  data = read_data(test_set)
  len_data = data.shape[0]
  output = np.zeros((len_data, T))
  
  for i in range(0, len_data, max_batch_size):
    batch_size = min(max_batch_size, len_data-i)
  
    #run process block
    process_net.blobs['rand_data'].reshape(batch_size, T, max_value)
    if max_value > 1:
      process_net.blobs['rand_data'].data[...] = mat_to_one_hot(data[i:i+batch_size,...], max_value)
    else:
      process_net.blobs['rand_data'].data[:,:,0] = data[i:i+batch_size,...]
    #initialize size for internal LSTM layers
    if 'q_init' in process_net.blobs.keys():
      process_net.blobs['q_init'].reshape(1, batch_size, message_size)
    else:
      process_net.blobs['q_star_0'].reshape(1, batch_size, message_size*2)
    process_net.blobs['c_0'].reshape(1, batch_size, message_size)
    process_net.blobs['x'].reshape(1, batch_size, message_size*2)
    process_net.blobs['cont_1'].reshape(1, batch_size)
    if 'cont_all' in process_net.blobs.keys():
      process_net.blobs['cont_all'].reshape(1, batch_size)
      process_net.blobs['cont_all'].data[...] = np.ones((1, batch_size))
    process_net.forward()
  
    #run write block
    write_net.blobs['LSTM_input_0'].reshape(batch_size, 1, max_value)
    write_net.blobs['message'].reshape(batch_size, T, message_size)
    write_net.blobs['q_star_T'].reshape(1, batch_size, message_size*2) #q_star_T is the cell statefrom previous block
    write_net.blobs['gen_c_0'].reshape(1, batch_size, message_size*2)
    write_net.blobs['cont_1'].reshape(1, batch_size)
  
  
    LSTM_input_0 = np.zeros((batch_size, 1, max_value))
    message = copy.deepcopy(process_net.blobs['read_ip_0'].data) 
    q_star_T = copy.deepcopy(process_net.blobs['q_star_%d' %num_process_steps].data) 
    gen_c_0 = np.zeros((1, batch_size, message_size*2))
    cont_0 = np.ones((1, batch_size))
  
    for t in range(T):
      write_net.blobs['LSTM_input_0'].data[...] = LSTM_input_0
      write_net.blobs['message'].data[...] = message
      write_net.blobs['q_star_T'].data[...] = q_star_T
      write_net.blobs['gen_c_0'].data[...] = gen_c_0
      write_net.blobs['cont_1'].data[...] = cont_0
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
        output[i:i+batch_size,t] = copy.deepcopy(LSTM_input_0.squeeze())
  
  incorrect = np.sum(np.abs(np.sum(np.sort(data, axis=1) - output, axis=1)) > 1e-5)
  accuracy = 1-(float(incorrect)/output.shape[0])
  print "Accuracy: %f" %(accuracy)
  print "Example outputs:"
  for i in range(10):
    print i, ": ", output[i,:]
  return accuracy

def set2seq_train(solver_path):
  solver = caffe.get_solver(solver_path)
  solver.solve()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--len_sequence", type=int, default=10) 
  parser.add_argument("--message_dim", type=int, default=10)
  parser.add_argument("--process_steps", type=int, default=1)
  parser.add_argument("--num_tests", type=int, default=1)
  args = parser.parse_args()
 
  #build net
  print "Building model..."
  tag = build_set2seq (args.len_sequence, 100, 1, args.message_dim, args.process_steps)
  accuracies = np.zeros((args.num_tests)) 
  for iter in range(args.num_tests):
    #train_net
    solver_path = 'prototxts/%s_solver.prototxt' %tag
    set2seq_train(solver_path)
    #test_net
    accuracies[iter] = test_set2seq(tag, args.len_sequence, args.message_dim, args.process_steps)

  avg_accuracy = np.mean(accuracies)
  std_accuracy = np.std(accuracies)
  print "For len_sequence: %d, message_dim: %d, process_steps: %d" %(args.len_sequence, args.message_dim, args.process_steps)
  print "Avg accuracy (%d): %f" %(args.num_tests, avg_accuracy)
  print "Std accuracy (%d): %f" %(args.num_tests, std_accuracy)

