import sys
sys.path.insert(0, '../../python/')
import caffe
import random
import numpy as np
from threading import Thread
import pdb

class sortDataGenerator(object):
 
  def __init__(self, len_sequence, max_value, batch_size, thread_result):
    self.len_sequence = len_sequence
    self.max_value = max_value
    self.batch_size = batch_size
    self.thread_result = thread_result

  def __call__(self):
    rand_mat = np.random.rand(self.len_sequence, self.batch_size)
    rand_mat = np.array(rand_mat*self.max_value, dtype=int)
    label_mat = np.sort(rand_mat, axis=0)
    self.thread_result['rand_mat'] = rand_mat
    self.thread_result['label_mat'] = label_mat

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
    self.top_names = ['rand_mat', 'label_mat']

    self.batchAdvancer = sortDataGenerator(self.params['len_sequence'], self.params['max_value'], 
                                           self.params['batch_size'], self.thread_result)
    self.dispatch_worker()
    self.join_worker()

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    for top_index, name in enumerate(self.top_names):
      shape = (self.len_sequence, self.batch_size)
      top[top_index].reshape(*shape)








