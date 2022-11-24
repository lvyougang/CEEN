# from sentence_em import HAN_Model
# from enviroment import ENV
from itertools import chain
import sys
import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
import pdb

class Policy(nn.Module):
    def __init__(self, hidden_size,batch_size, gru_size, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear11 = nn.Linear(gru_size, hidden_size)
        self.linear21 = nn.Linear(hidden_size, num_outputs)
        self.linear12 = nn.Linear(gru_size, hidden_size)
        self.linear22 = nn.Linear(hidden_size, num_outputs)
        self.linear13 = nn.Linear(gru_size, hidden_size)
        self.linear23 = nn.Linear(hidden_size, num_outputs)
        self.linear14 = nn.Linear(gru_size, hidden_size)
        self.linear24 = nn.Linear(hidden_size, num_outputs)
        self.batch_size=batch_size
        self.gru1 = nn.GRUCell(input_size=2 * gru_size, hidden_size=gru_size)
        self.gru2 = nn.GRUCell(input_size=2 * gru_size, hidden_size=gru_size)
        self.gru3 = nn.GRUCell(input_size=2 * gru_size, hidden_size=gru_size)
        self.gru4 = nn.GRUCell(input_size=2 * gru_size, hidden_size=gru_size)
        self.h1=nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, gru_size)), requires_grad=True)
        self.h2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, gru_size)), requires_grad=True)
        self.h3 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, gru_size)), requires_grad=True)
        self.h4 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, gru_size)), requires_grad=True)


    def forward(self, inputs1,inputs2,inputs3,inputs4,start,h1_in=None,h2_in=None,h3_in=None,h4_in=None):

        if start:
            h1=self.h1.expand(self.batch_size,self.h1.shape[1])
            h2 = self.h2.expand(self.batch_size, self.h2.shape[1])
            h3 = self.h3.expand(self.batch_size, self.h3.shape[1])
            h4 = self.h4.expand(self.batch_size, self.h4.shape[1])
        else:
            h1=h1_in
            h2=h2_in
            h3=h3_in
            h4=h4_in
        h1_out=self.gru1(inputs1,h1)
        h2_out = self.gru2(inputs2, h2)
        h3_out = self.gru3(inputs3, h3)
        h4_out = self.gru4(inputs4, h4)


        inputs1 = F.relu(self.linear11(h1_out))
        action_scores1 = self.linear21(inputs1)

        inputs2 = F.relu(self.linear12(h2_out))
        action_scores2 = self.linear22(inputs2)

        inputs3 = F.relu(self.linear13(h3_out))
        action_scores3 = self.linear23(inputs3)

        inputs4 = F.relu(self.linear14(h4_out))
        action_scores4 = self.linear24(inputs4)


        return F.softmax(action_scores1,dim=1),F.softmax(action_scores2,dim=1),F.softmax(action_scores3,dim=1),F.softmax(action_scores4,dim=1),h1_out,h2_out,h3_out,h4_out
def select_action(model, state1,state2,state3,state4):
    # batch 8* sentence 10 *embedding
    state1=torch.transpose(state1, 0, 1)
    state2 = torch.transpose(state2, 0, 1)
    state3 = torch.transpose(state3, 0, 1)
    state4 = torch.transpose(state4, 0, 1)
    action_1 = []
    action_2 = []
    action_3 = []
    action_4 = []

    probs_1=[]
    probs_2 = []
    probs_3 = []
    probs_4 = []
    for j in range(state1.shape[0]):
        if j==0:
            outs = model(state1[j], state2[j], state3[j], state4[j],True)
        else:
            outs = model(state1[j], state2[j], state3[j], state4[j],False,h1_in,h2_in,h3_in,h4_in)
        probs1 = torch.clamp(outs[0], 1e-10, 1.0)
        probs2 = torch.clamp(outs[1], 1e-10, 1.0)
        probs3 = torch.clamp(outs[2], 1e-10, 1.0)
        probs4 = torch.clamp(outs[3], 1e-10, 1.0)

        probs_1.append(probs1)
        probs_2.append(probs2)
        probs_3.append(probs3)
        probs_4.append(probs4)


        h1_in = outs[4]
        h2_in = outs[5]
        h3_in = outs[6]
        h4_in = outs[7]
        action_1.append(probs1.multinomial(1).data)
        action_2.append(probs2.multinomial(1).data)
        action_3.append(probs3.multinomial(1).data)
        action_4.append(probs4.multinomial(1).data)

    action_1=torch.cat(action_1).reshape(state1.shape[1],state1.shape[0],-1)
    action_2 = torch.cat(action_2).reshape(state1.shape[1],state1.shape[0],-1)
    action_3 = torch.cat(action_3).reshape(state1.shape[1],state1.shape[0],-1)
    action_4 = torch.cat(action_4).reshape(state1.shape[1],state1.shape[0],-1)
    # batch*sentence_num*1

    probs_1=torch.cat(probs_1).reshape(state1.shape[1],state1.shape[0],-1)
    probs_2 = torch.cat(probs_2).reshape(state1.shape[1],state1.shape[0],-1)
    probs_3 = torch.cat(probs_3).reshape(state1.shape[1],state1.shape[0],-1)
    probs_4 = torch.cat(probs_4).reshape(state1.shape[1],state1.shape[0],-1)

    #batch*sentence_num*2


    prob1=torch.gather(probs_1,dim=2,index=action_1)
    prob2 = torch.gather(probs_2, dim=2, index=action_2)
    prob3 = torch.gather(probs_3, dim=2, index=action_3)
    prob4 = torch.gather(probs_4, dim=2, index=action_4)

    log_prob1 = prob1.log()
    log_prob2 = prob2.log()
    log_prob3 = prob3.log()
    log_prob4 = prob4.log()

    return action_1,action_2,action_3,action_4,log_prob1,log_prob2,log_prob3,log_prob4

def update_parameters(reward1,reward2,reward3,reward4, log_prob1,log_prob2,log_prob3,log_prob4, gamma,optimizer,scheduler):

    if True:
        R1 = torch.zeros(reward1.shape[0]).cuda()
        R2 = torch.zeros(reward2.shape[0]).cuda()
        R3 = torch.zeros(reward3.shape[0]).cuda()
        R4 = torch.zeros(reward4.shape[0]).cuda()
    else:
        R1 = torch.zeros(reward1.shape[0])
        R2 = torch.zeros(reward2.shape[0])
        R3 = torch.zeros(reward3.shape[0])
        R4 = torch.zeros(reward4.shape[0])


    reward1=reward1.squeeze(2).t()
    reward2 = reward2.squeeze(2).t()
    reward3 = reward3.squeeze(2).t()
    reward4 = reward4.squeeze(2).t()
    log_prob1 = log_prob1.squeeze(2).t()
    log_prob2 = log_prob2.squeeze(2).t()
    log_prob3 = log_prob3.squeeze(2).t()
    log_prob4 = log_prob4.squeeze(2).t()

    optimizer.zero_grad()
    total_loss=0
    R11=[]
    R21=[]
    R31=[]
    R41=[]

    for i in reversed(range(reward1.shape[0])):
        R1 = gamma * R1 + reward1[i]
        R2 = gamma * R2 + reward2[i]
        R3 = gamma * R3 + reward3[i]
        R4 = gamma * R4 + reward4[i]
        R11.append(R1)
        R21.append(R2)
        R31.append(R3)
        R41.append(R4)
    R11=torch.cat(R11,dim=0)
    R11=(R11-R11.mean())/R11.std()
    R11=R11.reshape(reward1.shape[0],-1)

    R21 = torch.cat(R21, dim=0)
    R21 = (R21 - R21.mean()) / R21.std()
    R21 = R21.reshape(reward1.shape[0], -1)

    R31 = torch.cat(R31, dim=0)
    R31 = (R31 - R31.mean()) / R31.std()
    R31 = R31.reshape(reward1.shape[0], -1)

    R41=torch.cat(R41,dim=0)
    R41=(R41-R41.mean())/R41.std()
    R41=R41.reshape(reward1.shape[0],-1)



    for i in reversed(range(reward1.shape[0])):
        # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss1 =  - (log_prob1[i] * R11[i]).sum()
        loss2 = - (log_prob2[i] * R21[i]).sum()
        loss3 =  - (log_prob3[i] * R31[i]).sum()
        loss4 =  - (log_prob4[i] * R41[i]).sum()
        loss=loss1+loss2+loss3+loss4
        total_loss+=loss

    total_loss.backward()
    utils.clip_grad_norm_(self.model.parameters(), 5)
    optimizer.step()
    scheduler.step()

    return total_loss.item(),((R1+R2+R3+R4)/(4)).mean().item()
class REINFORCE:
    def __init__(self, hidden_size, batch_size,gru_size, action_space,gpu=False):
        self.action_space = action_space
        self.model = Policy(hidden_size=hidden_size,batch_size=batch_size, gru_size=gru_size, action_space=action_space)

        self.model=torch.nn.DataParallel(self.model)

        self.gpu=gpu
        if gpu:
            self.model.cuda()
            # self.model1.cuda()
            # self.model2.cuda()
            # self.model3.cuda()
            # self.model4.cuda()
        # self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=32)
        self.scheduler_lr = StepLR(self.optimizer, step_size=2, gamma=0.1)

    def select_action(self, state1,state2,state3,state4):
        # probs = self.model(Variable(state).cuda())
        # batch 8* sentence 10 *embedding
        state1=torch.transpose(state1, 0, 1)
        state2 = torch.transpose(state2, 0, 1)
        state3 = torch.transpose(state3, 0, 1)
        state4 = torch.transpose(state4, 0, 1)
        action_1 = []
        action_2 = []
        action_3 = []
        action_4 = []

        probs_1=[]
        probs_2 = []
        probs_3 = []
        probs_4 = []
        for j in range(state1.shape[0]):
            if j==0:
                outs = self.model(state1[j], state2[j], state3[j], state4[j],True)
            else:
                outs = self.model(state1[j], state2[j], state3[j], state4[j],False,h1_in,h2_in,h3_in,h4_in)
            probs1 = torch.clamp(outs[0], 1e-10, 1.0)
            probs2 = torch.clamp(outs[1], 1e-10, 1.0)
            probs3 = torch.clamp(outs[2], 1e-10, 1.0)
            probs4 = torch.clamp(outs[3], 1e-10, 1.0)

            probs_1.append(probs1)
            probs_2.append(probs2)
            probs_3.append(probs3)
            probs_4.append(probs4)


            h1_in = outs[4]
            h2_in = outs[5]
            h3_in = outs[6]
            h4_in = outs[7]
            action_1.append(probs1.multinomial(1).data)
            action_2.append(probs2.multinomial(1).data)
            action_3.append(probs3.multinomial(1).data)
            action_4.append(probs4.multinomial(1).data)

        action_1=torch.cat(action_1).reshape(state1.shape[1],state1.shape[0],-1)
        action_2 = torch.cat(action_2).reshape(state1.shape[1],state1.shape[0],-1)
        action_3 = torch.cat(action_3).reshape(state1.shape[1],state1.shape[0],-1)
        action_4 = torch.cat(action_4).reshape(state1.shape[1],state1.shape[0],-1)
        # batch*sentence_num*1

        probs_1=torch.cat(probs_1).reshape(state1.shape[1],state1.shape[0],-1)
        probs_2 = torch.cat(probs_2).reshape(state1.shape[1],state1.shape[0],-1)
        probs_3 = torch.cat(probs_3).reshape(state1.shape[1],state1.shape[0],-1)
        probs_4 = torch.cat(probs_4).reshape(state1.shape[1],state1.shape[0],-1)

        #batch*sentence_num*2


        prob1=torch.gather(probs_1,dim=2,index=action_1)
        prob2 = torch.gather(probs_2, dim=2, index=action_2)
        prob3 = torch.gather(probs_3, dim=2, index=action_3)
        prob4 = torch.gather(probs_4, dim=2, index=action_4)

        log_prob1 = prob1.log()
        log_prob2 = prob2.log()
        log_prob3 = prob3.log()
        log_prob4 = prob4.log()

        return action_1,action_2,action_3,action_4,log_prob1,log_prob2,log_prob3,log_prob4

    def update_parameters(self,reward1,reward2,reward3,reward4, log_prob1,log_prob2,log_prob3,log_prob4, gamma):
        self.model.train()
        if self.gpu:
            R1 = torch.zeros(reward1.shape[0]).cuda()
            R2 = torch.zeros(reward2.shape[0]).cuda()
            R3 = torch.zeros(reward3.shape[0]).cuda()
            R4 = torch.zeros(reward4.shape[0]).cuda()
        else:
            R1 = torch.zeros(reward1.shape[0])
            R2 = torch.zeros(reward2.shape[0])
            R3 = torch.zeros(reward3.shape[0])
            R4 = torch.zeros(reward4.shape[0])

        reward1=reward1.squeeze(2).t()
        reward2 = reward2.squeeze(2).t()
        reward3 = reward3.squeeze(2).t()
        reward4 = reward4.squeeze(2).t()
        log_prob1 = log_prob1.squeeze(2).t()
        log_prob2 = log_prob2.squeeze(2).t()
        log_prob3 = log_prob3.squeeze(2).t()
        log_prob4 = log_prob4.squeeze(2).t()

        self.optimizer.zero_grad()
        total_loss=0
        R11=[]
        R21=[]
        R31=[]
        R41=[]

        for i in reversed(range(reward1.shape[0])):
            R1 = gamma * R1 + reward1[i]
            R2 = gamma * R2 + reward2[i]
            R3 = gamma * R3 + reward3[i]
            R4 = gamma * R4 + reward4[i]
            R11.append(R1)
            R21.append(R2)
            R31.append(R3)
            R41.append(R4)
        R11=torch.cat(R11,dim=0)
        R11=(R11-R11.mean())/R11.std()
        R11=R11.reshape(reward1.shape[0],-1)

        R21 = torch.cat(R21, dim=0)
        R21 = (R21 - R21.mean()) / R21.std()
        R21 = R21.reshape(reward1.shape[0], -1)

        R31 = torch.cat(R31, dim=0)
        R31 = (R31 - R31.mean()) / R31.std()
        R31 = R31.reshape(reward1.shape[0], -1)

        R41=torch.cat(R41,dim=0)
        R41=(R41-R41.mean())/R41.std()
        R41=R41.reshape(reward1.shape[0],-1)



        for i in reversed(range(reward1.shape[0])):
            # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
            loss1 =  - (log_prob1[i] * R11[i]).sum()
            loss2 = - (log_prob2[i] * R21[i]).sum()
            loss3 =  - (log_prob3[i] * R31[i]).sum()
            loss4 =  - (log_prob4[i] * R41[i]).sum()
            loss=loss1+loss2+loss3+loss4
            total_loss+=loss

        total_loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()

        return total_loss.item(),((R1+R2+R3+R4)/(4)).mean().item()

    def save(self,path):
        torch.save(self.model.state_dict(),path)



if __name__ == "__main__":
    han_model = HAN_Model(vocab_size=100, embedding_size=200, gru_size=20, dropoutp=0.5)
    han_model.eval()
    x = torch.Tensor(np.zeros([8, 10, 20])).long()
    x[0][0][0:10] = 1
    output = han_model(x,x)
    agent = REINFORCE(hidden_size=10,batch_size=8,gru_size=20,action_space=2)
    agent_prediction=agent.select_action(output[0],output[1],output[2],output[3])
    env= ENV(gru_size=20, dropoutp=0.5,class_num1=4,class_num2=4,class_num3=4,element_num1=4,element_num2=4,element_num3=4)
    env.eval()
    output_reward=env(output[0],output[1],output[2],output[3],output[4],agent_prediction[0],agent_prediction[1],agent_prediction[2],agent_prediction[3],output[5],output[6],output[7],output[8])
    loss,reward=agent.update_parameters(output_reward[6],output_reward[7],output_reward[8],output_reward[9], agent_prediction[4],agent_prediction[5],agent_prediction[6],agent_prediction[7], 1)
    print("{:.2f} {:.2f}".format(loss,reward))