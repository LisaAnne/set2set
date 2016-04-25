from build_net.caffe_net import *

class set2sequence_net(caffe_net):
  
  def __init__(self, param_str):
    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.param_str = param_str
    self.dim_q_star = param_str['len_sequence']*2
    self.N = param_str['batch_size']
    self.dim_m = param_str['len_sequence']
    self.dim_s = param_str['len_sequence']
    #hidden unit will be 2*self.dim_m, cell unit will be self.dim_m
    self.gate_dim = (self.dim_m)*4
    self.process_steps = 1 
    self.pointer_embed = 5
    self.in_dim = param_str['max_value']

  def build_read_block(self, in_x, layers=1):
    #in x should be n' X n where n' is a variable length set size and n is batch size
    last_layer = in_x
    for layer in range(layers):
      ip_name = 'read_ip_%d' %layer
      self.n.tops[ip_name] = L.InnerProduct(self.n.tops[last_layer], 
                                            num_output=self.dim_m,
                                            weight_filler=self.gaussian_filler(),
                                            bias_filler=self.gaussian_filler(1),
                                            param=self.learning_params([[1,1],[2,0]]),
                                            axis=2)
      last_layer = ip_name 
    return last_layer

  def sum_channel(self, message_set, message_shape):
    self.n.tops[message_set+'_reshape'] = L.Reshape(self.n.tops[message_set], 
                                                    shape=dict(dim=message_shape+[1]))
    self.n.tops[message_set+'_sum_int'] = L.Convolution(self.n.tops[message_set+'_reshape'],
                                                    kernel_size=1, stride=1,
                                                    num_output=1, pad=0, bias=False,
                                                    weight_filler=self.constant_filler(1),
                                                    param=self.learning_params([[0,0]]))
    return L.Reshape(self.n.tops[message_set+'_sum_int'],
                     shape=dict(dim=[1, message_shape[0], message_shape[2]]))

  def init_q_star0(self, message):
    self.n.tops['q_init'] = self.dummy_data_layer([1, self.N, self.dim_m], 0)
    self.n.tops['norm_message'] = L.Power(self.n.tops[message], scale=1./self.dim_m)
    self.n.tops['r_0'] = self.sum_channel('norm_message', 
                                          [-1, self.dim_s, self.dim_m])
    self.n.tops['q_star_0'] = L.Concat(self.n.tops['q_init'], self.n.tops['r_0'], axis=2) 


  def f(self, q, m, tag):
    self.n.tops[q + '_reshape'] = L.Reshape(self.n.tops[q],
                                           shape=dict(dim=[-1, self.dim_m, 1]))
    self.n.tops[q + '_tile'] = L.Tile(self.n.tops[q+'_reshape'], tiles=self.dim_s, axis=2)
    self.n.tops['e_int_%s' %tag] = L.Eltwise(self.n.tops[q+'_tile'], self.n.tops[m], 
                                             operation=0)
    return L.Reduction(self.n.tops['e_int_%s' %tag], axis=2)
     

  def build_process_block(self, message, process_steps):
    #build q_star0

    self.init_q_star0(message)
    self.n.tops['c_0'] = self.dummy_data_layer([1, self.N, self.dim_m], 0)
    self.n.tops['x'] = self.dummy_data_layer([1, self.N, self.dim_m + self.dim_m], 0)
    self.n.tops['cont_0'] = self.dummy_data_layer([1,self.N], 0)
    self.n.tops['cont_all'] = self.dummy_data_layer([1,self.N], 1)

    for t in range(process_steps):    
      q_star_tm1 = 'q_star_%d' %t
      c_tm1 = 'c_%d' %t
      e_t = 'e_%d' %(t+1)
      a_t = 'a_%d' %(t+1)
      a_t_reshape = 'a_%d_reshape' %(t+1)
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
      self.n.tops[e_t] = self.f(q_t, message, t)
      self.n.tops[a_t] = self.softmax(self.n.tops[e_t], axis=1)
      self.n.tops[a_t_reshape] = L.Reshape(self.n.tops[a_t],
                                           shape=dict(dim=[-1, self.dim_s, 1]))
      self.n.tops[a_t_tile] = L.Tile(self.n.tops[a_t_reshape], tiles=self.dim_m, axis=2)
      self.n.tops[r_t_mult] = L.Eltwise(self.n.tops[a_t_tile], self.n.tops[message], operation=0) 
      self.n.tops[r_t] = self.sum_channel(r_t_mult, 
                                          [-1, self.dim_s, self.dim_m]) 
      self.n.tops[q_star_t] = L.Concat(self.n.tops[q_t], self.n.tops[r_t], axis=2)  

    self.silence(self.n.tops[c_t])
    return q_star_t

  def build_write_pointer(self, LSTM_input, pointer_hidden, encode_vector, len_decoder, num_LSTM_inputs=None):
    #want pointer hidden to be N X S X D and encoder vector to be 1 X N X D

    if not num_LSTM_inputs: num_LSTM_inputs = self.dim_s
    if num_LSTM_inputs > 1:
      LSTM_inputs = L.Slice(self.n.tops[LSTM_input], axis=1, ntop=num_LSTM_inputs)
      self.rename_tops(LSTM_inputs, ['LSTM_input_%d' %d for d in range(self.dim_s)]) 
    else:
      self.rename_tops([self.n.tops[LSTM_input]], ['LSTM_input_0']) 
      
    self.n.tops['gen_c_0'] = self.dummy_data_layer([1, self.N, self.dim_m*2], 0)
    self.gate_dim *= 2

    if 'cont_0' not in self.n.tops.keys():
      self.n.tops['cont_0'] = self.dummy_data_layer([1,self.N], 0)
 
 
    #could have done this all without unrolling LSTM (I think?)
    for t in range(len_decoder):
      h_tm1 = 'gen_h_%d' %t   
      c_tm1 = 'gen_c_%d' %t   
      if t==0:
        h_tm1 = encode_vector
      x_in = 'LSTM_input_%d' %t
      x_t = 'LSTM_input_%d_reshape' %t   
      h_t = 'gen_h_%d' %(t+1)   
      c_t = 'gen_c_%d' %(t+1)
      en_t = 'encoder_%d' %(t+1)
      de_t = 'decoder_%d' %(t+1)
      de_t_reshape = 'decoder_reshape_%d' %(t+1)
      de_t_tile = 'decoder_tile_%d' %(t+1)
      sum_t = 'sum_en_de_%d' %(t+1)
      tanh_t = 'tanh_%d' %(t+1)
      u_t = 'u_%d' %(t+1)
      out_t = 'out_%d' %(t+1)
      if t > 0:
        cont = 'cont_all'
      else:
        cont = 'cont_0'

      self.n.tops[x_t] = L.Reshape(self.n.tops[x_in], shape=dict(dim=[1,-1, self.in_dim])) 
      #LSTM
      self.n.tops[h_t], self.n.tops[c_t] = self.lstm_unit('gen_lstm', self.n.tops[x_t], 
                                                          self.n.tops[cont], 
                                                          h=self.n.tops[h_tm1], 
                                                          c=self.n.tops[c_tm1],
                                                          timestep=t)
      #compute u
      self.n.tops[en_t] = L.InnerProduct(self.n.tops[pointer_hidden],
                                              num_output=self.pointer_embed,
                                              weight_filler= self.gaussian_filler(),
                                              bias_filler= self.constant_filler(1),
                                              param=self.named_params(['E_w', 'E_b'], 
                                                                      [[1,1],[2,0]]),
                                              axis=2)
      self.n.tops[de_t] = L.InnerProduct(self.n.tops[h_t],
                                              num_output=self.pointer_embed,
                                              bias=False,
                                              weight_filler= self.gaussian_filler(),
                                              param=self.named_params(['D_w'], [[1,1]]),
                                              axis=2)
      self.n.tops[de_t_reshape] = L.Reshape(self.n.tops[de_t], 
                                            shape=dict(dim=[-1, 1, self.pointer_embed]))
      self.n.tops[de_t_tile] = L.Tile(self.n.tops[de_t_reshape], tiles=self.dim_s, 
                                           axis=1)
      self.n.tops[sum_t] = L.Eltwise(self.n.tops[en_t], self.n.tops[de_t_tile], 
                                     operation=1)
      self.n.tops[tanh_t] = L.TanH(self.n.tops[sum_t])
      self.n.tops[u_t] = L.InnerProduct(self.n.tops[tanh_t], num_output=1,
                                        weight_filler=self.gaussian_filler(),
                                        bias_filler=self.constant_filler(1),
                                        param=self.named_params(['u_w', 'u_b'], [[1,1,],[2,0]]),
                                        axis=2)
      #compute output prob as u
      #self.n.tops[out_t] = L.Softmax(self.n.tops[u_t], axis=1) 

    self.gate_dim /= 2

  def build_set2seq(self, param_str, save_name):
    self.python_input_layer('python_layers', 'generateSortData', self.param_str)
    message = self.build_read_block('rand_data')
    q_star_t = self.build_process_block(message, self.process_steps)
    self.build_write_pointer('label_data', message, q_star_t, self.dim_s)
    self.silence(self.n.tops['gen_c_%d' %(self.dim_s)])
    self.n.tops['u_concat'] = L.Concat(*[self.n.tops['u_%d' %t] for t in range(1, self.dim_s+1)], axis=2)
    self.n.tops['loss'] = self.softmax_loss(self.n.tops['u_concat'], 
                                            self.n.tops['train_label_data'],
                                            axis=1,loss_weight=1)
#    self.n.tops['loss'] = self.softmax_loss(self.n.tops['u_concat'], 
#                                            self.n.tops['train_label_data'],
#                                            axis=2,loss_weight=1)
    self.write_net('prototxts/' + save_name)

    #build deploy net
    self.n = caffe.NetSpec()
    self.n.tops['rand_data'] = self.dummy_data_layer([self.N, self.dim_s, self.in_dim], 0)
    message = self.build_read_block('rand_data')
    q_star_t = self.build_process_block(message, self.process_steps)
    self.write_net('prototxts/deploy_process_' + save_name)

    self.n = caffe.NetSpec()
    self.n.tops['label_data'] = self.dummy_data_layer([self.N, 1, self.in_dim], 0)
    self.n.tops['message'] = self.dummy_data_layer([self.N, self.dim_s, self.dim_m], 0)
    self.n.tops['q_star_T'] = self.dummy_data_layer([1, self.N, 2*self.dim_s], 0)
    self.build_write_pointer('label_data', 'message', 'q_star_T', 1, num_LSTM_inputs=1)
    self.n.tops['probs'] = self.softmax(self.n.tops['u_1'], axis=1)
    #self.n.tops['probs'] = self.softmax(self.n.tops['u_1'], axis=2)
    self.write_net('prototxts/deploy_write_' + save_name)


