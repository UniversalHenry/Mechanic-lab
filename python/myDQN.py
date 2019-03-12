"""
!!!NOTICE!!!
From the CUMCM2018 participants:
This code is modified by our team to solve the B problem in CUMCM2018, mainly used the DQN class,
but changed the structure of the network and also use environment written by ourselves according to our model
From the origin author:
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
numpy
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math as M

random_seed = 0
for SetOrder in range(1,4):
    for ProcessNum in range(1,3):
        for ContainErr in range(2):
            EpochNum = 120

            np.random.seed(random_seed)
            torch.manual_seed(np.random.randint(random_seed,random_seed+100))


            def loaddata(setorder):
                if setorder == 1:
                    RGVtime1step,RGVtime2step,RGVtime3step, \
                    CNCtime1in1,CNCtime1in2,CNCtime2in2, \
                    RGVtimeOddCNC,RGVtimeEvenCNC,RGVtimeclean \
                    = 20,33,46,560,400,378,28,31,25
                elif setorder == 2:
                    RGVtime1step,RGVtime2step,RGVtime3step, \
                    CNCtime1in1,CNCtime1in2,CNCtime2in2, \
                    RGVtimeOddCNC,RGVtimeEvenCNC,RGVtimeclean \
                    =23,41,59,580,280,500,30,35,30
                elif setorder == 3:
                    RGVtime1step,RGVtime2step,RGVtime3step, \
                    CNCtime1in1,CNCtime1in2,CNCtime2in2, \
                    RGVtimeOddCNC,RGVtimeEvenCNC,RGVtimeclean \
                    =18,41,59,580,280,500,30,35,30

                return [RGVtime1step,RGVtime2step,RGVtime3step], \
                       [CNCtime1in1,CNCtime1in2,CNCtime2in2], \
                       [RGVtimeOddCNC,RGVtimeEvenCNC,RGVtimeclean]

            work_time = 8*3600

            def get_err_pro(time_1run):
                err_pro_1time = 0.01
                err_pro_1s = 1 - (1-err_pro_1time)**(1/(time_1run))
                return err_pro_1s

            def get_fix_time():
                pi = 3.1415926
                delte = 0.148082075
                while 1:
                    x = np.random.rand() * 10
                    y = np.random.rand()
                    y_ =1 / ((2 * pi) ** 0.5 * delte * x) * M.exp(-(np.log(x) - np.log(5)) ** 2 / (2 * delte ** 2))
                    if y < y_:
                        return int((x + 10)*60)

            def get_err(randdata,serial,err_pro):
                return  randdata[serial] < err_pro

            class CNC:
                def __init__(self,num_process,contain_err,CNCtime):
                    self.timer = 0
                    self.thing = 0
                    self.thingorder = 0
                    if num_process == 1:
                        self.processtime = [CNCtime[0]]
                    if num_process == 2:
                        self.processtime = [CNCtime[1],CNCtime[2]]
                    if contain_err:
                        self.err = [get_err_pro(x) for x in self.processtime]
                        self.randdata = np.random.rand(work_time).tolist()

            class RGV:
                def __init__(self,setorder,num_process,contain_err):
                    self.timer = 0
                    self.place = 0
                    self.action = 0     #0,1,2,3 represent move to the place, 4 hold odd, 5 hold even ,6 clean
                    self.thing = 0
                    self.thingorder = 0
                    RGVtime1,CNCtime,RGVtime2 = loaddata(setorder)
                    self.steptime = [0,RGVtime1[0],RGVtime1[1],RGVtime1[2]]
                    self.holdtime = [RGVtime2[1],RGVtime2[0]]
                    self.cleantime = RGVtime2[2]
                    self.cnc = [CNC(num_process,contain_err,CNCtime) for i in range(8)]

            class Env:                  #our environment
                def __init__(self, setorder, num_process, contain_err):
                    self.action_space = 7
                    self.observation_space = 18
                    self.timer = 0
                    self.setorder = setorder
                    self.numprocess = num_process
                    self.containerr = contain_err
                    self.thingorder = 0
                    self.onrecord = {'cnc':[],'time':[],'thingorder':[],'process':[]}
                    self.offrecord = {'cnc':[],'time':[],'thingorder':[],'process':[]}
                    self.rgv = RGV(setorder,num_process,contain_err)
                    self.rewardaccumulate = 0
                    self.punishaccumulate = 0
                    self.reward = 0
                    self.punish = 0
                    if contain_err:
                        self.recerr = {'cnc':[],'time':[]}

                def reset(self):
                    r1 = self.rewardaccumulate
                    r2 = self.punishaccumulate
                    self.__init__(self.setorder,self.numprocess,self.containerr)
                    self.rewardaccumulate = r1
                    self.punishaccumulate = r2
                    return [0 for i in range(18)]

                def counter(self):
                    self.timer += 1
                    if self.rgv.timer > 0:
                        self.rgv.timer -= 1
                    cnctimer1 = [self.rgv.cnc[i].timer for i in range(8)]
                    for i in range(8):
                        if self.rgv.cnc[i].timer > 0:
                            self.rgv.cnc[i].timer -= 1
                        if cnctimer1[i] != 0 and self.rgv.cnc[i].timer == 0:
                            self.rgv.cnc[i].thing += 1
                    punish = cnctimer1.count(0)
                    self.punish += punish
                    self.punishaccumulate += punish
                    if self.containerr:
                        for i in range(8):
                            if self.rgv.cnc[i].timer>0:
                                if get_err(self.rgv.cnc[i].randdata,min(self.timer,work_time-1),self.rgv.cnc[i].err[self.rgv.cnc[i].thing]):
                                    self.rgv.cnc[i].thing = -1
                                    self.rgv.cnc[i].thingorder = 0
                                    self.rgv.cnc[i].timer = get_fix_time()
                                    self.recerr['cnc'] += [i]
                                    self.recerr['time'] += [self.timer]


                def step(self,action):
                    assert self.rgv.timer == 0
                    self.rgv.action = action
                    if action>=0 and action<=3:
                        self.rgv.timer = self.rgv.steptime[abs(self.rgv.place-action)]
                        self.rgv.place = action

                    if action==4:               #even in here cnc 1,3,5,7 represent CNC #2,4,6,8
                        if self.rgv.cnc[self.rgv.place*2+1].timer == 0 and self.rgv.cnc[self.rgv.place*2+1].thing >= 0:
                            tmpthing = self.rgv.thing
                            self.rgv.thing = self.rgv.cnc[self.rgv.place*2+1].thing
                            self.rgv.cnc[self.rgv.place*2+1].thing = tmpthing
                            self.rgv.timer = self.rgv.holdtime[0]
                            tmpthingorder = self.rgv.thingorder
                            self.rgv.thingorder = self.rgv.cnc[self.rgv.place*2+1].thingorder
                            self.rgv.cnc[self.rgv.place*2+1].thingorder = tmpthingorder
                            if tmpthing == 0:
                                self.thingorder += 1
                                self.rgv.cnc[self.rgv.place*2+1].thingorder = self.thingorder
                                self.onrecord['cnc'].append(self.rgv.place*2+2)
                                self.onrecord['time'].append(self.timer)
                                self.onrecord['thingorder'].append(self.thingorder)
                                self.onrecord['process'].append(1)
                            if tmpthing > 0:
                                self.onrecord['cnc'].append(self.rgv.place*2+2)
                                self.onrecord['time'].append(self.timer)
                                self.onrecord['thingorder'].append(self.thingorder)
                                self.onrecord['process'].append(tmpthing+1)
                            if self.rgv.thing > 0:
                                self.offrecord['cnc'].append(self.rgv.place*2+2)
                                self.offrecord['time'].append(self.timer)
                                self.offrecord['thingorder'].append(self.thingorder)
                                self.offrecord['process'].append(tmpthing)
                            if self.rgv.cnc[self.rgv.place*2+1].processtime.__len__() > tmpthing:
                                self.rgv.cnc[self.rgv.place*2+1].timer = self.rgv.cnc[self.rgv.place*2+1].processtime[tmpthing] + self.rgv.holdtime[0]
                        else:
                            self.punish += 1
                            self.punishaccumulate +=1

                    if action==5:               #even in here cnc 0,2,4,6 represent CNC #1,3,5,7
                        if self.rgv.cnc[self.rgv.place*2].timer == 0 and self.rgv.cnc[self.rgv.place*2].thing >= 0:
                            tmpthing = self.rgv.thing
                            self.rgv.thing = self.rgv.cnc[self.rgv.place*2].thing
                            self.rgv.cnc[self.rgv.place*2].thing = tmpthing
                            self.rgv.timer = self.rgv.holdtime[1]
                            tmpthingorder = self.rgv.thingorder
                            self.rgv.thingorder = self.rgv.cnc[self.rgv.place*2].thingorder
                            self.rgv.cnc[self.rgv.place*2].thingorder = tmpthingorder
                            if tmpthing == 0:
                                self.thingorder += 1
                                self.rgv.cnc[self.rgv.place*2].thingorder = self.thingorder
                                self.onrecord['cnc'].append(self.rgv.place*2+1)
                                self.onrecord['time'].append(self.timer)
                                self.onrecord['thingorder'].append(self.thingorder)
                                self.onrecord['process'].append(1)
                            if tmpthing > 0:
                                self.onrecord['cnc'].append(self.rgv.place*2+1)
                                self.onrecord['time'].append(self.timer)
                                self.onrecord['thingorder'].append(self.thingorder)
                                self.onrecord['process'].append(tmpthing+1)
                            if self.rgv.thing > 0:
                                self.offrecord['cnc'].append(self.rgv.place*2+1)
                                self.offrecord['time'].append(self.timer)
                                self.offrecord['thingorder'].append(self.thingorder)
                                self.offrecord['process'].append(tmpthing)
                            if self.rgv.thing < self.numprocess:
                                self.reward += (self.rgv.thing+1)**2
                                self.rewardaccumulate += (self.rgv.thing+1)**2
                            if self.rgv.cnc[self.rgv.place*2].processtime.__len__() > tmpthing:
                                self.rgv.cnc[self.rgv.place*2].timer = self.rgv.cnc[self.rgv.place*2].processtime[tmpthing] + self.rgv.holdtime[1]
                        else:
                            self.punish += 1
                            self.punishaccumulate +=1

                    if action==6:
                        if self.rgv.thing == self.numprocess:
                            self.rgv.timer = self.rgv.cleantime
                            self.rgv.thingorder = 0
                            self.rgv.thing = 0
                            self.reward += (self.rgv.thing+1)**2
                            self.rewardaccumulate += (self.rgv.thing +1)**2
                        else:
                            self.punish += 1
                            self.punishaccumulate +=1

                    s_ = [self.rgv.cnc[i].timer/ max(self.rgv.cnc[0].processtime) for i in range(8)]
                    s_ = s_+ [(self.rgv.cnc[i].thing+ 1) /(self.numprocess +1) for i in range(8)]
                    s_ = s_ + [self.rgv.place / 3,self.rgv.thing /self.numprocess]

                    if self.timer > work_time:
                        done = 1
                    else:
                        done = 0

                    r1 = self.reward*self.punishaccumulate/max(self.rewardaccumulate,0.01)
                    r2 = self.punish*self.rewardaccumulate/max(self.punishaccumulate,0.01)
                    r = 1/(1+np.exp(r2 - r1))*2-1

                    env.reward = 0
                    env.punish = 0

                    if self.containerr:
                        info = {'onrecord':self.onrecord,'offrecord':self.offrecord,'recerr':self.recerr}
                    else:
                        info = {'onrecord':self.onrecord,'offrecord':self.offrecord}
                    return s_, r, done, info


            env = Env(SetOrder,ProcessNum,ContainErr)        # environment ( setorder, num_process, contain_err)

            # Hyper Parameters
            BATCH_SIZE = 32
            LR = 0.01                   # learning rate
            EPSILON = 0.9               # greedy policy
            GAMMA = 0.9                 # reward discount
            TARGET_REPLACE_ITER = 100   # target update frequency
            MEMORY_CAPACITY = 2000
            N_ACTIONS = env.action_space
            N_STATES = env.observation_space

            class Net(nn.Module):
                def __init__(self, ):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(N_STATES, 100)
                    self.fc1.weight.data.normal_(0, 0.1)   # initialization
                    self.fc2 = nn.Linear(100, 150)
                    self.fc2.weight.data.normal_(0, 0.1)   # initialization
                    self.fc3 = nn.Linear(150, 100)
                    self.fc3.weight.data.normal_(0, 0.1)   # initialization
                    self.out = nn.Linear(100, N_ACTIONS)
                    self.out.weight.data.normal_(0, 0.1)   # initialization

                def forward(self, x):
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.fc2(x)
                    x = F.relu(x)
                    x = self.fc3(x)
                    x = F.relu(x)
                    actions_value = self.out(x)
                    return actions_value


            class DQN(object):
                def __init__(self):
                    self.eval_net, self.target_net = Net(), Net()
                    self.learn_step_counter = 0                                     # for target updating
                    self.memory_counter = 0                                         # for storing memory
                    self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
                    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
                    self.loss_func = nn.MSELoss()

                def choose_action(self, x):
                    x = torch.unsqueeze(torch.FloatTensor(x), 0)
                    # input only one sample
                    if np.random.uniform() < EPSILON:   # greedy
                        actions_value = self.eval_net.forward(x)
                        action = torch.max(actions_value, 1)[1].data.numpy()
                        action = action[0]                                          # return the argmax index
                    else:   # random
                        action = np.random.randint(0, N_ACTIONS)
                    return action

                def val_choose_action(self, x):
                    x = torch.unsqueeze(torch.FloatTensor(x), 0)
                    actions_value = self.eval_net.forward(x)
                    action = torch.max(actions_value, 1)[1].data.numpy()
                    action = action[0]                                          # return the argmax index
                    return action

                def store_transition(self, s, a, r, s_):
                    transition = np.hstack((s, [a, r], s_))
                    # replace the old memory with new memory
                    index = self.memory_counter % MEMORY_CAPACITY
                    self.memory[index, :] = transition
                    self.memory_counter += 1

                def learn(self):
                    # target parameter update
                    if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                        self.target_net.load_state_dict(self.eval_net.state_dict())
                    self.learn_step_counter += 1

                    # sample batch transitions
                    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
                    b_memory = self.memory[sample_index, :]
                    b_s = torch.FloatTensor(b_memory[:, :N_STATES])
                    b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
                    b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
                    b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

                    # q_eval w.r.t the action in experience
                    q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
                    q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
                    q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
                    loss = self.loss_func(q_eval, q_target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            dqn = DQN()

            rec = []
            valrec = []
            info = {}
            valinfo = {}
            print('\nCollecting experience...')
            for i_episode in range(EpochNum):
                print('(episode'+str(i_episode)+'/'+str(EpochNum)+')')
                s = env.reset()
                while True:
                    if env.rgv.timer == 0:
                        a = dqn.choose_action(s)
                        # take action
                        s_, r, done, info = env.step(a)
                        dqn.store_transition(s, a, r, s_)
                        if done:
                            break
                        if dqn.memory_counter > MEMORY_CAPACITY:
                            dqn.learn()
                        s = s_
                    env.counter()
                rec = rec + [max(info['offrecord']['thingorder']+[0])]
                print(str('train_product_num:'+str(max(info['offrecord']['thingorder']+[0]))))
                mark = '('+str(SetOrder)+'_'+str(ProcessNum)+'_'+str(ContainErr)+')'
                if not os.path.exists('./'+mark):
                    os.mkdir('./'+mark)
                if rec.index(max(rec)) == i_episode:
                    torch.save({'target_net':dqn.target_net,'eval_net':dqn.eval_net},'./'+mark+'/best_dqn'+mark+'.tar.gz')
                    torch.save({'info':info,'rec':rec,'best':max(rec)},'./'+mark+'/info'+mark+'.tar.gz')
                if (i_episode+1) % 5 == 0:
                    torch.save({'target_net':dqn.target_net,'eval_net':dqn.eval_net},'./'+mark+'/dqn'+str(i_episode)+'ep'+mark+'.tar.gz')
                    torch.save({'info':info,'rec':rec,'best':max(rec)},'./'+mark+'/info'+mark+'.tar.gz')

                # test
                s = env.reset()
                while True:
                    if env.rgv.timer == 0:
                        a = dqn.val_choose_action(s)
                        # take action
                        s_, r, done, val_info = env.step(a)
                        if done:
                            break
                        s = s_
                    env.counter()
                valrec = valrec + [max(val_info['offrecord']['thingorder']+[0])]
                print(str('val_product_num:'+str(max(val_info['offrecord']['thingorder']+[0]))))
                mark = '('+str(SetOrder)+'_'+str(ProcessNum)+'_'+str(ContainErr)+')'
                if valrec.index(max(valrec)) == i_episode:
                    torch.save({'target_net':dqn.target_net,'eval_net':dqn.eval_net},'./'+mark+'/val_best_dqn'+mark+'.tar.gz')
                    torch.save({'info':info,'rec':rec,'best':max(rec)},'./'+mark+'/val_info'+mark+'.tar.gz')
