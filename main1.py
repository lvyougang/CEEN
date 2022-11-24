import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle as pk
import numpy as np
import os
import random
import time
import argparse
from Model.han_LSTM_relu import HAN_Model
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics

#设置config参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--max_epoch', type=int, default=30, help='max epoch')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
#通过命令行输入使用的gpu



model_save='checkpoint/'

current_time=int(time.time())
print("current_time:{}".format(current_time))
#输出模型开始时间，为查找对应保存模型文件
#评价函数
def evaluation_multitask(y, prediction, task_num):
    metrics_acc = []
    for x in range(task_num):
        accuracy_metric = metrics.accuracy_score(y[x], prediction[x])
        macro_recall = metrics.recall_score(y[x], prediction[x], average='macro')
        macro_precision = metrics.precision_score(y[x], prediction[x], average='macro')
        macro_f1 = metrics.f1_score(y[x], prediction[x], average='macro')
        metrics_acc.append(
            (accuracy_metric, macro_precision, macro_recall, macro_f1))
    return metrics_acc

#固定随机初始化


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed = args.seed

loss_rate=1
#模型超参数
batch_size = args.batch_size
cuda=True
max_epoch = args.max_epoch

sent_len_fact = 100
doc_len_fact = 15

learning_rate = 1e-5
lstm_size = 128

task = ['law ', 'accu', 'time']


#词典加载
with open('data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()

emb_path = 'data/cail_thulac.npy'
word_embedding = np.cast[np.float32](np.load(emb_path))


word_dict_len = len(word2id_dict)

vec_size = 200
shuffle = True

n_law = 103
n_accu = 119
n_term = 12
n_keti=75
n_zhuti=7
n_zhuguan=2
n_keguan=63
#模型构建

model = HAN_Model(vocab_size=word_dict_len,
                  embedding_size=vec_size,
                  gru_size=lstm_size, dropoutp=args.dropout,class_num1=n_law,class_num2=n_accu,class_num3=n_term,element_num1=n_keti,element_num2=n_zhuti,element_num3=n_zhuguan,element_num4=n_keguan,
                  weights=word_embedding, is_pretrain=True)

model = torch.nn.DataParallel(model)

#多gpu
#m模型加载cuda config
if cuda and torch.cuda.is_available():
    model.cuda()


#设置loss函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=32)


#读入数据
f_train = pk.load(open('data/train_criminal_element4.pkl', 'rb'))
f_valid = pk.load(open('data/valid_criminal_element4.pkl', 'rb'))
f_test = pk.load(open('data/test_criminal_element4.pkl', 'rb'))

train_step = int(len(f_train['fact_list']) / batch_size) + 1
lose_num_train = train_step * batch_size - len(f_train['fact_list'])

valid_step = int(len(f_valid['fact_list']) / batch_size) + 1
lose_num_valid = valid_step * batch_size - len(f_valid['fact_list'])

test_step = int(len(f_test['fact_list']) / batch_size) + 1
lose_num_test = test_step * batch_size - len(f_test['fact_list'])

fact_train = f_train['fact_list']
law_labels_train = f_train['law_label_lists']
accu_label_train = f_train['accu_label_lists']
term_train = f_train['term_lists']
keti=f_train['keti']
zhuti=f_train['zhuti']
zhuguan=f_train['zhuguan']
keguan=f_train['keguan']
#记录最好的结果
best_law=0
best_accu=0
best_term=0
best_epoch=0
if shuffle:
    index = [i for i in range(len(f_train['term_lists']))]
    random.shuffle(index)
    fact_train = [fact_train[i] for i in index]
    law_labels_train = [law_labels_train[i] for i in index]
    accu_label_train = [accu_label_train[i] for i in index]
    term_train = [term_train[i] for i in index]
    keti = [keti[i] for i in index]
    zhuti = [zhuti[i] for i in index]
    zhuguan = [zhuguan[i] for i in index]
    keguan = [keguan[i] for i in index]
    for epoch in range(max_epoch):
        total_loss=0
        for i in range(train_step):
            if i == train_step - 1:
                inputs = np.array(fact_train[i * batch_size:] + fact_train[:lose_num_train])
                law_labels_input = np.array(law_labels_train[i * batch_size:] + law_labels_train[:lose_num_train])
                accu_labels_input = np.array(accu_label_train[i * batch_size:] + accu_label_train[:lose_num_train])
                time_labels_input = np.array(term_train[i * batch_size:] + term_train[:lose_num_train])
                keti_input=np.array(keti[i * batch_size:] + keti[:lose_num_train])
                zhuti_input = np.array(zhuti[i * batch_size:] + zhuti[:lose_num_train])
                zhuguan_input = np.array(zhuguan[i * batch_size:] + zhuguan[:lose_num_train])
                keguan_input = np.array(keguan[i * batch_size:] + keguan[:lose_num_train])
            else:
                inputs = np.array(fact_train[i * batch_size: (i + 1) * batch_size])
                law_labels_input = np.array(law_labels_train[i * batch_size: (i + 1) * batch_size])
                accu_labels_input = np.array(accu_label_train[i * batch_size: (i + 1) * batch_size])
                time_labels_input = np.array(term_train[i * batch_size: (i + 1) * batch_size])
                keti_input = np.array(keti[i * batch_size: (i + 1) * batch_size])
                zhuti_input = np.array(zhuti[i * batch_size: (i + 1) * batch_size])
                zhuguan_input = np.array(zhuguan[i * batch_size: (i + 1) * batch_size])
                keguan_input = np.array(keguan[i * batch_size: (i + 1) * batch_size])
            # print(inputs.shape)

            model.train()
            # mask
            fact_mask = torch.from_numpy(inputs - word2id_dict['BLANK'])
            #输入转换成张量
            inputs=torch.from_numpy(inputs)
            law_labels_input = torch.from_numpy(law_labels_input)
            # print(law_labels_input[0])
            accu_labels_input = torch.from_numpy(accu_labels_input)
            time_labels_input = torch.from_numpy(time_labels_input)
            keti_input = torch.from_numpy(keti_input)
            zhuti_input = torch.from_numpy(zhuti_input)
            zhuguan_input = torch.from_numpy(zhuguan_input)
            keguan_input = torch.from_numpy(keguan_input)
            #one-hot标签
            # law_labels_input=torch.nn.functional.one_hot(law_labels_input, n_law)
            # accu_labels_input = torch.nn.functional.one_hot(accu_labels_input, n_accu)
            # time_labels_input = torch.nn.functional.one_hot(time_labels_input, n_term)


            ##需要修改的地方
            if cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                law_labels_input = law_labels_input.cuda()
                accu_labels_input = accu_labels_input.cuda()
                time_labels_input = time_labels_input.cuda()

                keti_input = keti_input.cuda()
                zhuti_input = zhuti_input.cuda()
                zhuguan_input = zhuguan_input.cuda()
                keguan_input = keguan_input.cuda()

                fact_mask = fact_mask.cuda()
            if cuda and torch.cuda.is_available():
                out = model(inputs,fact_mask, gpu=True)
            else:
                out = model(inputs,fact_mask)
            loss1 = criterion(out[0],law_labels_input.long())
            loss2 = criterion(out[1], accu_labels_input.long())
            loss3 = criterion(out[2], time_labels_input.long())
            loss4=criterion(out[3], keti_input.long())
            loss5 = criterion(out[4], zhuti_input.long())
            loss6 = criterion(out[5], zhuguan_input.long())
            loss7=criterion(out[6], keguan_input.long())

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6+loss7
            total_loss+=loss
            #计算loss
            optimizer.zero_grad()
            #梯度清0
            loss.backward()
            #反向传播
            optimizer.step()
            scheduler.step()
            #更新参数
        print('epoch: {} loss: {:.2f} '.format(epoch+1,total_loss/train_step))


        # ############################----------the following is valid prediction-----------------###############################
        predic_law, predic_accu, predic_time = [], [], []

        for i in range(valid_step):
            if i == valid_step - 1:
                inputs = np.array(f_valid['fact_list'][i * batch_size:] + f_valid['fact_list'][:lose_num_valid])
                law_labels_input = np.array(f_valid['law_label_lists'][i * batch_size:] + f_valid['law_label_lists'][:lose_num_valid])
                accu_labels_input = np.array(f_valid['accu_label_lists'][i * batch_size:] + f_valid['accu_label_lists'][:lose_num_valid])
                time_labels_input = np.array(f_valid['term_lists'][i * batch_size:] + f_valid['term_lists'][:lose_num_valid])
            else:
                inputs = np.array(f_valid['fact_list'][i * batch_size: (i + 1) * batch_size])
                law_labels_input = np.array(f_valid['law_label_lists'][i * batch_size: (i + 1) * batch_size])
                accu_labels_input = np.array(f_valid['accu_label_lists'][i * batch_size: (i + 1) * batch_size])
                time_labels_input = np.array(f_valid['term_lists'][i * batch_size: (i + 1) * batch_size])
            model.eval()
            # mask
            fact_mask = torch.from_numpy(inputs - word2id_dict['BLANK'])
            # 输入转换成张量
            inputs = torch.from_numpy(inputs)
            law_labels_input = torch.from_numpy(law_labels_input)
            # print(law_labels_input[0])
            accu_labels_input = torch.from_numpy(accu_labels_input)
            time_labels_input = torch.from_numpy(time_labels_input)

            ##需要修改的地方
            if cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                law_labels_input = law_labels_input.cuda()
                accu_labels_input = accu_labels_input.cuda()
                time_labels_input = time_labels_input.cuda()
                fact_mask = fact_mask.cuda()
            if cuda and torch.cuda.is_available():
                out = model(inputs, fact_mask, gpu=True)
            else:
                out = model(inputs, fact_mask)

            predic_law += list((torch.argmax(out[0], 1)).cpu().numpy())
            predic_accu += list((torch.argmax(out[1], 1)).cpu().numpy())
            predic_time += list((torch.argmax(out[2], 1)).cpu().numpy())

        prediction = [predic_law, predic_accu, predic_time]
        y = [f_valid['law_label_lists']+ f_valid['law_label_lists'][:lose_num_valid], f_valid['accu_label_lists'] + f_valid['accu_label_lists'][:lose_num_valid], f_valid['term_lists']+ f_valid['term_lists'][:lose_num_valid]]
        metric = evaluation_multitask(y, prediction, 3)
        print("valid:")
        for i in range(3):

            print('{}: Acc={:.2f}% MP={:.2f}% MR={:.2f}% F1={:.2f}%'.format(task[i],metric[i][0]*100,metric[i][1]*100,metric[i][2]*100,metric[i][3]*100))


        ############################----------the following is test prediction-----------------###############################
        predic_law, predic_accu, predic_time = [], [], []
        for i in range(test_step):
            if i == test_step - 1:
                inputs = np.array(f_test['fact_list'][i * batch_size:] + f_test['fact_list'][:lose_num_test])
                law_labels_input = np.array(
                    f_test['law_label_lists'][i * batch_size:] + f_test['law_label_lists'][:lose_num_test])
                accu_labels_input = np.array(
                    f_test['accu_label_lists'][i * batch_size:] + f_test['accu_label_lists'][:lose_num_test])
                time_labels_input = np.array(
                    f_test['term_lists'][i * batch_size:] + f_test['term_lists'][:lose_num_test])
            else:
                inputs = np.array(f_test['fact_list'][i * batch_size: (i + 1) * batch_size])
                law_labels_input = np.array(f_test['law_label_lists'][i * batch_size: (i + 1) * batch_size])
                accu_labels_input = np.array(f_test['accu_label_lists'][i * batch_size: (i + 1) * batch_size])
                time_labels_input = np.array(f_test['term_lists'][i * batch_size: (i + 1) * batch_size])
            model.eval()
            # mask
            fact_mask = torch.from_numpy(inputs - word2id_dict['BLANK'])
            # 输入转换成张量
            inputs = torch.from_numpy(inputs)
            law_labels_input = torch.from_numpy(law_labels_input)
            accu_labels_input = torch.from_numpy(accu_labels_input)
            time_labels_input = torch.from_numpy(time_labels_input)

            ##需要修改的地方
            if cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                law_labels_input = law_labels_input.cuda()
                accu_labels_input = accu_labels_input.cuda()
                time_labels_input = time_labels_input.cuda()
                fact_mask = fact_mask.cuda()
            if cuda and torch.cuda.is_available():
                out = model(inputs, fact_mask, gpu=True)
            else:
                out = model(inputs, fact_mask)

            predic_law += list((torch.argmax(out[0], 1)).cpu().numpy())
            predic_accu += list((torch.argmax(out[1], 1)).cpu().numpy())
            predic_time += list((torch.argmax(out[2], 1)).cpu().numpy())
        print('test:')
        prediction = [predic_law, predic_accu, predic_time]
        y = [f_test['law_label_lists']+ f_test['law_label_lists'][:lose_num_test], f_test['accu_label_lists']+ f_test['accu_label_lists'][:lose_num_test], f_test['term_lists']+ f_test['term_lists'][:lose_num_test]]
        metric = evaluation_multitask(y, prediction, 3)
        print("{:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}% {:.2f}%".format(metric[0][0] * 100,
                                                                            metric[0][1] * 100, metric[0][2] * 100,
                                                                            metric[0][3] * 100,metric[1][0] * 100,
                                                                            metric[1][1] * 100, metric[1][2] * 100,
                                                                            metric[1][3] * 100,metric[2][0] * 100,
                                                                            metric[2][1] * 100, metric[2][2] * 100,
                                                                            metric[2][3] * 100))
