import sys
sys.path.insert(0, '../../python/')
import caffe
import random
import numpy as np
from threading import Thread
import pdb
from python_utils import *

class sortDataRead(object):

  def __init__(self, data, batch_size, max_value, thread_result):
    self.data = data
    self.n = data.shape[0]
    self.len_sequence = self.data.shape[1]
    self.iteration = 0
    self.thread_result = thread_result
    self.batch_size = batch_size
    self.max_value = max_value

  def __call__(self): 
    rand_mat = np.zeros((self.batch_size, self.len_sequence))
    if self.iteration + self.batch_size >= self.n:
      rand_mat[:self.n-self.iteration,:] = self.data[self.iteration:self.n,:]
      rand_mat[self.n-self.iteration:,:] = self.data[:self.n-self.iteration,:]
      self.iteration = self.n-self.iteration
    else:
      rand_mat = self.data[self.iteration:self.iteration+self.batch_size,:]
      self.iteration += self.batch_size

    label_mat = np.sort(rand_mat, axis=1)
    train_label_mat = np.argsort(rand_mat, axis=1)
    rand_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    label_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    a1_idx = [[i]*self.len_sequence for i in range(self.batch_size)]    
    a1_idx = [i for j in a1_idx for i in j]
    a2_idx = range(self.len_sequence)*self.batch_size
    rand_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(rand_mat)] = 1
    label_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(label_mat)] = 1

    label_one_hot_mat_shift = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    label_one_hot_mat_shift[:,1:,:] = label_mat_one_hot[:,:-1,:]

    self.thread_result['rand_mat'] = rand_mat_one_hot
    self.thread_result['label_mat'] = label_one_hot_mat_shift
    self.thread_result['train_label_mat'] = train_label_mat #train_label_mat.reshape((self.batch_size, self.len_sequence, 1))

class sortDataGeneratorOne(object):
 
  def __init__(self, len_sequence, max_value, batch_size, thread_result):
    self.len_sequence = len_sequence
    self.max_value = max_value
    self.batch_size = batch_size
    self.thread_result = thread_result
    self.write_txt = open('train_generate_sents.txt', 'w')
    self.write_txt.writelines('begin\n')
    self.write_txt.close()

  def __call__(self):
    self.write_txt = open('train_generate_sents.txt', 'a')
    rand_mat = np.random.rand(self.batch_size, self.len_sequence)
    #rand_mat = np.array(rand_mat*1000, dtype=int)
    label_mat = np.sort(rand_mat, axis=1)
    train_label_mat = np.argsort(rand_mat, axis=1)

    label_shift = np.zeros((self.batch_size, self.len_sequence))
    label_shift[:,1:] = label_mat[:,:-1]

    self.thread_result['rand_mat'] = rand_mat.reshape((self.batch_size, self.len_sequence, 1))
    self.thread_result['label_mat'] = label_shift.reshape((self.batch_size, self.len_sequence, 1))
    #self.thread_result['train_label_mat'] = train_label_mat_one_hot
    self.thread_result['train_label_mat'] = train_label_mat #train_label_mat.reshape((self.batch_size, self.len_sequence, 1))
    for i in range(self.batch_size):
      self.write_txt.writelines('%s\n' %(' '.join([str(m) for m in np.ndarray.tolist(rand_mat[i,:])])))
    self.write_txt.close()

class sortDataGenerator(object):
 
  def __init__(self, len_sequence, max_value, batch_size, thread_result):
    self.len_sequence = len_sequence
    self.max_value = max_value
    self.batch_size = batch_size
    self.thread_result = thread_result

  def __call__(self):
    rand_mat = np.random.rand(self.batch_size, self.len_sequence)
    rand_mat = np.array(rand_mat*self.max_value, dtype=int)
    label_mat = np.sort(rand_mat, axis=1)
    train_label_mat = np.argsort(rand_mat, axis=1)
    rand_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    label_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    train_label_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.len_sequence))
    a1_idx = [[i]*self.len_sequence for i in range(self.batch_size)]    
    a1_idx = [i for j in a1_idx for i in j]
    a2_idx = range(self.len_sequence)*self.batch_size
    rand_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(rand_mat)] = 1
    label_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(label_mat)] = 1
    train_label_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(train_label_mat)] = 1

    label_one_hot_mat_shift = np.zeros((self.batch_size, self.len_sequence, self.max_value))
    label_one_hot_mat_shift[:,1:,:] = label_mat_one_hot[:,:-1,:]

    self.thread_result['rand_mat'] = rand_mat_one_hot
    self.thread_result['label_mat'] = label_one_hot_mat_shift
    #self.thread_result['train_label_mat'] = train_label_mat_one_hot
    self.thread_result['train_label_mat'] = train_label_mat #train_label_mat.reshape((self.batch_size, self.len_sequence, 1))

#  def __call__(self):
#    rand_mat = np.random.rand(self.batch_size, self.len_sequence)
#    rand_mat = np.array(rand_mat*self.max_value, dtype=int)
#    label_mat = np.sort(rand_mat, axis=1)
#    rand_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
#    label_mat_one_hot = np.zeros((self.batch_size, self.len_sequence, self.max_value))
#    a1_idx = [[i]*self.len_sequence for i in range(self.batch_size)]    
#    a1_idx = [i for j in a1_idx for i in j]
#    a2_idx = range(self.len_sequence)*self.batch_size
#    rand_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(rand_mat)] = 1
#    label_mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(label_mat)] = 1
#
#    self.thread_result['rand_mat'] = rand_mat_one_hot
#    self.thread_result['label_mat'] = label_mat_one_hot

class caffeDataLayer(caffe.Layer):

  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batchAdvancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def forward(self, bottom, top):
    if self.thread is not None:
      self.join_worker()

    for top_index, name in zip(range(len(top)), self.top_names):
      top[top_index].data[...] = self.thread_result[name] 
    self.dispatch_worker()

  def reshape(self, bottom, top):
    pass

  def backward(self, bottom, top):
    pass

class generateSortData(caffeDataLayer):

  def setup(self, bottom, top):

    self.params = eval(self.param_str)
    assert 'len_sequence' in self.params.keys()
    assert 'max_value' in self.params.keys()
    assert 'batch_size' in self.params.keys()

    self.len_sequence = self.params['len_sequence']
    self.max_value = self.params['max_value']
    self.batch_size = self.params['batch_size']

    self.thread_result = {}
    self.thread = None
    self.top_names = ['rand_mat', 'label_mat', 'train_label_mat']

    self.batchAdvancer = sortDataGenerator(self.params['len_sequence'], self.params['max_value'], 
                                           self.params['batch_size'], self.thread_result)
    self.dispatch_worker()
    self.join_worker()

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    for top_index, name in enumerate(self.top_names):
      if name == 'train_label_mat':
        #shape = (self.batch_size, self.len_sequence, 1)
        shape = (self.batch_size, self.len_sequence)
      else:
        shape = (self.batch_size, self.len_sequence, self.max_value)
      top[top_index].reshape(*shape)

class generateSortDataOne(caffeDataLayer):

  def setup(self, bottom, top):

    self.params = eval(self.param_str)
    assert 'len_sequence' in self.params.keys()
    assert 'max_value' in self.params.keys()
    assert 'batch_size' in self.params.keys()

    self.len_sequence = self.params['len_sequence']
    self.max_value = self.params['max_value']
    self.batch_size = self.params['batch_size']

    self.thread_result = {}
    self.thread = None
    self.top_names = ['rand_mat', 'label_mat', 'train_label_mat']

    self.batchAdvancer = sortDataGeneratorOne(self.params['len_sequence'], 
                                              self.params['max_value'], 
                                              self.params['batch_size'], 
                                              self.thread_result)
    self.dispatch_worker()
    self.join_worker()

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    for top_index, name in enumerate(self.top_names):
      if name == 'train_label_mat':
        #shape = (self.batch_size, self.len_sequence, 1)
        shape = (self.batch_size, self.len_sequence)
      else:
        shape = (self.batch_size, self.len_sequence, 1)
      top[top_index].reshape(*shape)

class readSortData(caffeDataLayer):

  def setup(self, bottom, top):

    self.params = eval(self.param_str)
    assert 'len_sequence' in self.params.keys()
    assert 'max_value' in self.params.keys()
    assert 'batch_size' in self.params.keys()

    self.len_sequence = self.params['len_sequence']
    self.max_value = self.params['max_value']
    self.batch_size = self.params['batch_size']

    data_txt = 'utils/data/ls_%d_mv_%d_train.txt' %(self.len_sequence, self.max_value)
    self.data = read_data(data_txt)

    self.thread_result = {}
    self.thread = None
    self.top_names = ['rand_mat', 'label_mat', 'train_label_mat']

    self.batchAdvancer = sortDataRead(self.data, self.params['batch_size'], 
                                      self.params['max_value'], self.thread_result)
    self.dispatch_worker()
    self.join_worker()

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    for top_index, name in enumerate(self.top_names):
      if name == 'train_label_mat':
        #shape = (self.batch_size, self.len_sequence, 1)
        shape = (self.batch_size, self.len_sequence)
      else:
        shape = (self.batch_size, self.len_sequence, self.max_value)
      top[top_index].reshape(*shape)







