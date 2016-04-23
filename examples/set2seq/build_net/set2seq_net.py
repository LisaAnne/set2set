from build_net.caffe_net import *

class set2sequence_net(caffe_net):
  
  def __init__(self, param_str):
    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.param_str = param_str
    self.dim_q_star = param_str['len_sequence']*2
    self.N = param_str['batch_size']
    self.dim_m = param_str['len_sequence']
    self.dim_r = param_str['len_sequence']
    self.dim_q = param_str['len_sequence']
    self.dim_c = param_str['len_sequence']
    self.gate_dim = self.dim_c*4
    self.process_steps = 1 

  def build_read_block(self, in_x, layers=1, hu=10):
    #in x should be n' X n where n' is a variable length set size and n is batch size
    last_layer = in_x
    for layer in range(layers):
      ip_name = 'read_ip_%d' %layer
      self.n.tops[ip_name] = L.InnerProduct(self.n.tops[last_layer], num_output=hu,
                                            weight_filler=self.gaussian_filler(),
                                            bias_filler=self.gaussian_filler(1),
                                            param=self.learning_params([[1,1],[2,0]]))
      last_layer = ip_name 
    return last_layer

  def init_q_star0(self, message):
    self.n.tops['q_init'] = self.dummy_data_layer([1, self.N, self.dim_q], 0)
    self.n.tops['norm_message'] = L.Power(self.n.tops[message], scale=1./self.dim_m)
    self.n.tops['r_0'] = L.Reduction(self.n.tops['norm_message'], axis=2)
    self.n.tops['q_star_0'] = L.Concat(self.n.tops['q_init'], self.n.tops['r_0'], axis=2) 

  def constant_init(self, shape, constant=0):
    return self.dummy_data_layer(shape, constant) 

  def f(self, x1, x2, tag):
    self.n.tops['e_int_%s' %tag] = L.Eltwise(x1, x2, operation=0)
    return L.Reduction(self.n.tops['e_int_%s' %tag], axis=2)
     

  def build_process_block(self, message, process_steps):
    #build q_star0

    self.init_q_star0(message)
    self.n.tops['c_0'] = self.constant_init([1, self.N, self.dim_c])
    self.n.tops['x'] = self.constant_init([1, self.N, self.dim_q + self.dim_m])
    self.n.tops['cont_0'] = self.constant_init([1,self.dim_q + self.dim_m], 0)
    self.n.tops['cont_all'] = self.constant_init([1,self.dim_q + self.dim_m], 1)

    for t in range(process_steps):    
      q_star_tm1 = 'q_star_%d' %t
      c_tm1 = 'c_%d' %t
      e_t = 'e_%d' %(t+1)
      a_t = 'a_%d' %(t+1)
      a_t_tile = 'a_%d_tile' %(t+1)
      r_t_mult = 'r_%d_mult' %(t+1)
      r_t = 'r_%d' %(t+1)
      q_star_t = 'q_star_%d' %(t+1)
      c_t = 'c_%d' %(t+1)
      q_t = 'q_%d' %(t+1)
      if t > 0:
        cont = 'cont_all'
      else:
        cont = 'cont_0'

      #lstm unit (input hidden unit, cell unit, x which is just all zeros)
      #returns h and c
      self.n.tops[q_t], self.n.tops[c_t] = self.lstm_unit('lstm', self.n.tops['x'], 
                                                          self.n.tops[cont], 
                                                          h=self.n.tops[q_star_tm1], 
                                                          c=self.n.tops[c_tm1],
                                                          timestep=t)

      #compute rt
#      self.n.tops[e_t] = self.f(self.n.tops[message], self.n.tops[q_t], t)
#      self.n.tops[a_t] = self.softmax(e_t, axis=2)
#      self.n.tops[a_t_tile] = L.Tile(self.n.tops[a_t], tiles=self.dim_m, axis=2)
#      self.n.tops[r_t_mult] = L.Eltwise(self.n.tops[a_t_tile], self.n.tops[message], operation=0) 
#      self.n.tops[r_t] = L.Reduction(self.n.tops[r_t_mult], axis=1) 
#      self.n.tops[q_star_t] = L.Concat([self.n.tops[q_t], self.n.tops[r_t]])  


    return q_star_t

  def build_set2seq(self, param_str, save_name):
    self.python_input_layer('python_layers', 'generateSortData', self.param_str)
    message = self.build_read_block('rand_data')
    q_star_t = self.build_process_block(message, self.process_steps)
    self.write_net('prototxts/' + save_name)
