import os
from torch import nn, optim
import torch.utils.data as Data
import pandas as pd
import cv2
from model import Transformer
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class NetDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(NetDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def getData(dataset):
    data_list = pd.read_csv(dataset)
    filenames = data_list['filename']
    pic_data = []
    dec_inputs = data_list['dec_inputs']
    dec_outputs = data_list['dec_outputs']
    suffix = ['FT_256_1.jpg','FT_512_1.jpg','FT_768_1.jpg','FT_1024_1.jpg','PT_1.jpg']
    for filename in filenames:
        data = []
        for suf in suffix:
            pic_read = cv2.imread('./output/figs/'+filename+suf)
            pic_read = cv2.cvtColor(pic_read, cv2.COLOR_BGR2GRAY) 
            data.append(pic_read)
        pic_read = cv2.imread('./output/figs/'+filename+'VT51.jpg')
        pic_read = cv2.cvtColor(pic_read, cv2.COLOR_BGR2GRAY) 
        pic_read = cv2.resize(pic_read,(640,480))
        data.append(pic_read)
        pic_data.append(data)
    with open('dict.json','r') as f:
        target_dict = json.load(f)
    for inputs in dec_inputs:
        words = inputs.split(' ')
        for word in words:
            if word not in target_dict:
                target_dict.append(word)
    for outputs in dec_outputs:
        words = outputs.split(' ')
        for word in words:
            if word not in target_dict:
                target_dict.append(word)
    with open('dict.json','w') as f:
        json.dump(target_dict,f)
    tgt_vocab = {target_dict[i]:i for i in range(len(target_dict))}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    dec_inputs_num = []
    dec_outputs_num = []
    for inputs in dec_inputs:
        dec_input = [tgt_vocab[n] for n in inputs.split()]
        dec_inputs_num.append(dec_input)
    for outputs in dec_outputs:
        dec_output = [tgt_vocab[n] for n in outputs.split()]
        dec_outputs_num.append(dec_output)
    return torch.FloatTensor(pic_data),torch.LongTensor(dec_inputs_num),torch.LongTensor(dec_outputs_num),idx2word
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
pic_data,dec_inputs,dec_outputs,idx2word = getData('./data_train.csv')
loader = Data.DataLoader(NetDataSet(pic_data, dec_inputs, dec_outputs), 1, True)

model = Transformer(len(idx2word)).cuda()
ctc_loss = nn.CTCLoss(reduction='mean')
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
print(model.parameters())
writer = SummaryWriter()
loss_value = []
for epoch in range(100):
    loss_batch = []
    for enc_inputs, dec_inputs, dec_outputs in loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #   outputs = torch.tensor(outputs,dtype=torch.float32)
    #  dec_inputs = torch.tensor(dec_inputs,dtype=torch.float32)
      mask = dec_outputs.view(-1)!=0
      mask_tensor = torch.masked_select(dec_outputs.view(-1), mask)
      loss = ctc_loss(F.log_softmax(outputs), mask_tensor,torch.tensor(10).type(torch.long),torch.tensor(len(mask_tensor)).type(torch.long))
      loss_batch.append(float(loss))
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      loss.requires_grad_(True)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    #  print([x.grad for x in optimizer.param_groups[0]['params']])
    writer.add_scalar('Loss/train', np.mean(loss_batch), epoch)
    loss_value.append(np.mean(loss_batch))
plt.plot(range(100),loss_value)
plt.show()