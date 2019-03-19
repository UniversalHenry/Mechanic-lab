import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from envModel import envModel
from init_nn import weight_init
import os
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

EPOCH = 200
CONTACT_TIMES = 5
EPOCH_SIZE = 500
INPUT_SIZE = 11 # [x1,x2,x3,v1,v2,v3,c2,contact times]
OUTPUT_SIZE = 30  # [m1,c1,k1] for 10 binary digits
LR = 0.01

f = open("./rec_v2.txt", 'w+')

# prepare predictor model
class myPredictor(nn.Module):
    def __init__(self):
        super(myPredictor, self).__init__()
        self.input_bn = nn.BatchNorm1d(INPUT_SIZE)
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=128,         # rnn hidden unit
            num_layers=10,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.output1 = nn.Linear(128, 128)
        self.output2 = nn.Linear(128, 256)
        self.output3 = nn.Linear(256, OUTPUT_SIZE)
        for m in self.modules():
            weight_init(m)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x=x.permute(0,2,1)
        x=self.input_bn(x)
        x =x.permute(0, 2, 1)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.output1(r_out[:, -1, :])
        out = nn.Sigmoid()(out)
        out = self.output2(out)
        out = nn.Sigmoid()(out)
        out = self.output3(out)
        # out = nn.Hardtanh(min_val=0.0,max_val=1.0)(out)
        return out

MP = myPredictor()
print(MP)
print(MP,file=f)
optimizer = torch.optim.Adam(MP.parameters(), lr=LR)
loss_func = nn.MSELoss()

# prepare data
Env = envModel()
def gen_data(p,stop):
    Env.gen(p=p,stop=stop)
    while True:
        if not Env.solve():
            break
    x = torch.tensor(Env.info['x'])
    v = torch.tensor(Env.info['v'])
    c = torch.tensor([[Env.c[1]] * x.shape[0]]).view(-1, 1)
    c1 = torch.tensor([[Env.c[0]] * x.shape[0]]).view(-1, 1)
    m1 = torch.tensor([[Env.m[0]] * x.shape[0]]).view(-1, 1)
    k1 = torch.tensor([[Env.k[0]] * x.shape[0]]).view(-1, 1)
    contact = torch.FloatTensor([Env.info['contact']]).view(-1, 1)
    condition = torch.cat((x, v, c, c1, m1, k1, contact), 1)
    condition = condition.view(1, condition.shape[0], condition.shape[1])
    c, k, m = Env.c_rand, Env.k_rand, Env.m_rand
    result = torch.zeros([1,OUTPUT_SIZE])
    for i in range(int(OUTPUT_SIZE/3)):
        c = c - int(c)
        k = k - int(k)
        m = m - int(m)
        if c >= 0.5:
            result[0][i] = 1.0
        if k >= 0.5:
            result[0][int(i+OUTPUT_SIZE/3)] = 1.0
        if m >= 0.5:
            result[0][int(i+OUTPUT_SIZE/3*2)] = 1.0
        c *= 2
        k *= 2
        m *= 2
    result = 2 * result - 1         # cast to -1 to 1   PAY ATTENTION!!!
    condition = Variable(condition)
    result = Variable(result)
    return condition,result


# training and testing
for epoch in range(EPOCH):
    for step in range(EPOCH_SIZE):
        condition,result = gen_data(p=False,stop=CONTACT_TIMES)
        predition = MP(condition)
        loss = loss_func(result, predition)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            p = (predition[0]>0).tolist()
            x, c, k, m = 1.0,0.0,0.0,0.0
            for i in range(int(OUTPUT_SIZE/3)):
                x = x/2
                c += float(p[i]) * x
                k += float(p[int(OUTPUT_SIZE/3)+i]) * x
                m += float(p[int(OUTPUT_SIZE/3)*2+i]) * x
            print('----------------------------------------------------------------------')
            print('Epoch: ', epoch, '\t| step: ', step, '\t| train loss: %.4f' % loss.data[0])
            print('c_rand groundtruth:%.4f\t|c_rand pred:%.4f\t|c_rand error:%.4f' % (Env.c_rand, c, abs(Env.c_rand - c)))
            print('k_rand groundtruth:%.4f\t|k_rand pred:%.4f\t|k_rand error:%.4f' % (Env.k_rand, k, abs(Env.k_rand - k)))
            print('m_rand groundtruth:%.4f\t|m_rand pred:%.4f\t|m_rand error:%.4f' % (Env.m_rand, m, abs(Env.m_rand - m)))
            print('----------------------------------------------------------------------',file=f)
            print('Epoch: ', epoch, '\t| step: ', step, '\t| train loss: %.4f' % loss.data[0],file=f)
            print('c_rand groundtruth:%.4f\t|c_rand pred:%.4f\t|c_rand error:%.4f' % (Env.c_rand, c, abs(Env.c_rand - c)),file=f)
            print('k_rand groundtruth:%.4f\t|k_rand pred:%.4f\t|k_rand error:%.4f' % (Env.k_rand, k, abs(Env.k_rand - k)),file=f)
            print('m_rand groundtruth:%.4f\t|m_rand pred:%.4f\t|m_rand error:%.4f' % (Env.m_rand, m, abs(Env.m_rand - m)),file=f)
    if not os.path.exists('./MP_v2'):
        os.mkdir('./MP_v2')
    torch.save(MP, './MP_v2/MP_epoch%d_v2.pkl' % (epoch+1))
