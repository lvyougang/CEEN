# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


class HAN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, gru_size, dropoutp,class_num1, class_num2,class_num3,element_num1,element_num2,element_num3,element_num4,is_pretrain=False, weights=None):
        super(HAN_Model, self).__init__()
        # 判断是否词向量是否预训练，不是预训练效果会差一点
        if is_pretrain:
            weights=torch.from_numpy(weights).float()
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 词的双向gru
        self.word_gru1 = nn.LSTM(input_size=embedding_size, hidden_size=gru_size, num_layers=1,
                               bidirectional=True, batch_first=True)
        self.word_gru2 = nn.LSTM(input_size=embedding_size, hidden_size=gru_size, num_layers=1,
                               bidirectional=True, batch_first=True)
        self.word_gru3 = nn.LSTM(input_size=embedding_size, hidden_size=gru_size, num_layers=1,
                               bidirectional=True, batch_first=True)
        self.word_gru4 = nn.LSTM(input_size=embedding_size, hidden_size=gru_size, num_layers=1,
                               bidirectional=True, batch_first=True)
        # 词attention的Query
        self.word_context1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        # print(self.word_context1)
        self.word_context2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.word_context3 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)

        self.word_context4 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        #multi-head
        self.word_dense1 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.word_dense2 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.word_dense3 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.word_dense4 = nn.Linear(2 * gru_size, 2 * gru_size)
        # 句子的双向gru，这里的输入shape是上一句的输出2*gru_size,
        self.sentence_gru1 = nn.LSTM(input_size=2 * gru_size, hidden_size=gru_size, num_layers=1,
                                   bidirectional=True, batch_first=True)
        self.sentence_gru2 = nn.LSTM(input_size=2 * gru_size, hidden_size=gru_size, num_layers=1,
                                   bidirectional=True, batch_first=True)
        self.sentence_gru3 = nn.LSTM(input_size=2 * gru_size, hidden_size=gru_size, num_layers=1,
                                   bidirectional=True, batch_first=True)
        self.sentence_gru4 = nn.LSTM(input_size=2 * gru_size, hidden_size=gru_size, num_layers=1,
                                   bidirectional=True, batch_first=True)
        # 句attention的Query
        self.sentence_context1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context3 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context4 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        # self.to_context4=nn.Linear(2 * gru_size, 2 * gru_size)
        #multi-head
        self.sentence_dense1 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.sentence_dense2 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.sentence_dense3 = nn.Linear(2 * gru_size, 2 * gru_size)
        self.sentence_dense4 = nn.Linear(2 * gru_size, 2 * gru_size)
        # class_num是最后文本分类的类别数量
        self.fc1 = nn.Linear(8 * gru_size, 256)
        self.fc2 = nn.Linear(8 * gru_size, 256)
        self.fc3 = nn.Linear(8 * gru_size, 256)
        self.fc11 = nn.Linear(256, class_num1)
        self.fc21 = nn.Linear(256, class_num2)
        self.fc31 = nn.Linear(256, class_num3)
        #要素类别
        self.ec1 = nn.Linear(2 * gru_size, 256)
        self.ec2 = nn.Linear(2 * gru_size, 256)
        self.ec3 = nn.Linear(2 * gru_size, 256)
        self.ec4=nn.Linear(2 * gru_size, 256)
        self.ec11 = nn.Linear(256, element_num1)
        self.ec21 = nn.Linear(256, element_num2)
        self.ec31 = nn.Linear(256, element_num3)
        self.ec41=nn.Linear(256, element_num4)
        # BN
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7= nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropoutp)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,x_mask, gpu=False):
        # 句子个数
        sentence_num = x.shape[1]
        # 句子长度
        sentence_length = x.shape[2]
        # x原来维度是bs*sentence_num*sentence_length，经过下面变化成二维的：(bs*sentence_num)*sentence_length
        x = x.view([-1, sentence_length])
        x_mask=x_mask.view([-1, sentence_length])
        # 加了embedding维度：(bs*sentence_num)*sentence_length*embedding_size
        x_embedding = self.embedding(x)
        # word_outputs.shape:(bs*sentence_num)*sentence_length*(2*gru_size)，这里因为是双向gru所以是2*gru_size
        x_embedding=self.dropout(x_embedding)
        word_outputs1, word_hidden = self.word_gru1(x_embedding)
        word_outputs2, word_hidden = self.word_gru2(x_embedding)
        word_outputs3, word_hidden = self.word_gru3(x_embedding)
        word_outputs4, word_hidden = self.word_gru4(x_embedding)
        #
        #word_hidden正向和反向最后输出的结果
        # attention_word_outputs.shape:(bs*sentence_num)*sentence_length*(2*gru_size)，这里对应原文公式5
        attention_word_outputs1 = torch.tanh(self.word_dense1(word_outputs1))
        attention_word_outputs2 = torch.tanh(self.word_dense2(word_outputs2))
        attention_word_outputs3 = torch.tanh(self.word_dense3(word_outputs3))
        attention_word_outputs4 = torch.tanh(self.word_dense4(word_outputs4))
        # weights.shape:(bs*sentence_num)*sentence_length*1，，这里和下面一句对应原文公式6

        weights1 = torch.matmul(attention_word_outputs1, self.word_context1)
        weights2 = torch.matmul(attention_word_outputs2, self.word_context2)
        weights3 = torch.matmul(attention_word_outputs3, self.word_context3)
        weights4 = torch.matmul(attention_word_outputs4, self.word_context4)

        # weights.shape: (bs * sentence_num) * sentence_length * 1
        weights1 = F.softmax(weights1, dim=1)
        weights2 = F.softmax(weights2, dim=1)
        weights3 = F.softmax(weights3, dim=1)
        weights4 = F.softmax(weights4, dim=1)
        # 这里对x加一个维度：(bs*sentence_num)*sentence_length*1
        x_mask_word=x_mask.unsqueeze(2)
        # 对x加的这个维度进行判断，如果这个维度上为0，表示这个地方是pad出来的，没有必要进行计算。
        # 如果为1，则按weight进行计算比例
        if gpu:
            weights1 = torch.where(x_mask_word != 0, weights1, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
            weights2 = torch.where(x_mask_word != 0, weights2, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
            weights3 = torch.where(x_mask_word != 0, weights3, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
            weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        else:
            weights1 = torch.where(x_mask_word != 0, weights1, torch.full_like(x_mask_word, 0, dtype=torch.float))
            weights2 = torch.where(x_mask_word != 0, weights2, torch.full_like(x_mask_word, 0, dtype=torch.float))
            weights3 = torch.where(x_mask_word != 0, weights3, torch.full_like(x_mask_word, 0, dtype=torch.float))
            weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float))


        # 这里由于忽略掉了pad为0的部分，所以要按维度重新计算分布
        weights1 = weights1 / (torch.sum(weights1, dim=1).unsqueeze(1) + 1e-4)
        weights2 = weights2 / (torch.sum(weights2, dim=1).unsqueeze(1) + 1e-4)
        weights3 = weights3 / (torch.sum(weights3, dim=1).unsqueeze(1) + 1e-4)
        weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)
        # bs*sentence_num*(2*gru_size)
        sentence_vector1 = torch.sum(word_outputs1 * weights1, dim=1).view([-1, sentence_num, word_outputs1.shape[-1]])
        sentence_vector2 = torch.sum(word_outputs2 * weights2, dim=1).view([-1, sentence_num, word_outputs2.shape[-1]])
        sentence_vector3 = torch.sum(word_outputs3 * weights3, dim=1).view([-1, sentence_num, word_outputs3.shape[-1]])
        sentence_vector4 = torch.sum(word_outputs4 * weights4, dim=1).view([-1, sentence_num, word_outputs4.shape[-1]])

        # sentence_outputs.shape:bs*sentence_num*(2*gru_size)
        sentence_outputs1, sentence_hidden = self.sentence_gru1(sentence_vector1)
        sentence_outputs2, sentence_hidden = self.sentence_gru2(sentence_vector2)
        sentence_outputs3, sentence_hidden = self.sentence_gru3(sentence_vector3)
        sentence_outputs4, sentence_hidden = self.sentence_gru4(sentence_vector4)
        # 对应原文公式8：bs*sentence_num*(2*gru_size)
        attention_sentence_outputs1 = torch.tanh(self.sentence_dense1(sentence_outputs1))
        attention_sentence_outputs2 = torch.tanh(self.sentence_dense2(sentence_outputs2))
        attention_sentence_outputs3 = torch.tanh(self.sentence_dense3(sentence_outputs3))
        attention_sentence_outputs4  = torch.tanh(self.sentence_dense4(sentence_outputs4))
        # bs*sentence_num*1
        weights1 = torch.matmul(attention_sentence_outputs1, self.sentence_context1)
        weights2 = torch.matmul(attention_sentence_outputs2, self.sentence_context2)
        weights3 = torch.matmul(attention_sentence_outputs3, self.sentence_context3)
        weights4 = torch.matmul(attention_sentence_outputs4, self.sentence_context4)

        # bs*sentence_num*1
        weights1 = F.softmax(weights1, dim=1)
        weights2 = F.softmax(weights2, dim=1)
        weights3 = F.softmax(weights3, dim=1)
        weights4 = F.softmax(weights4, dim=1)
        # bs * sentence_num * sentence_length
        x_mask_sentence = x_mask_word.view(-1, sentence_num, x_mask_word.shape[1])
        # bs * sentence_num * 1
        x_mask_sentence = torch.sum(x_mask_sentence, dim=2).unsqueeze(2)
        if gpu:
            # bs * sentence_num * 1
            weights1 = torch.where(x_mask_sentence != 0, weights1, torch.full_like(x_mask_sentence, 0, dtype=torch.float).cuda())
            weights2 = torch.where(x_mask_sentence != 0, weights2, torch.full_like(x_mask_sentence, 0, dtype=torch.float).cuda())
            weights3 = torch.where(x_mask_sentence != 0, weights3, torch.full_like(x_mask_sentence, 0, dtype=torch.float).cuda())
            weights4 = torch.where(x_mask_sentence != 0, weights4, torch.full_like(x_mask_sentence, 0, dtype=torch.float).cuda())
        else:
            weights1 = torch.where(x_mask_sentence != 0, weights1, torch.full_like(x_mask_sentence, 0, dtype=torch.float))
            weights2 = torch.where(x_mask_sentence != 0, weights2, torch.full_like(x_mask_sentence, 0, dtype=torch.float))
            weights3 = torch.where(x_mask_sentence != 0, weights3, torch.full_like(x_mask_sentence, 0, dtype=torch.float))
            weights4 = torch.where(x_mask_sentence != 0, weights4, torch.full_like(x_mask_sentence, 0, dtype=torch.float))
        # bs * sentence_num * 1，对未计算的pad进行缩放
        weights1 = weights1 / (torch.sum(weights1, dim=1).unsqueeze(1) + 1e-4)
        weights2 = weights2 / (torch.sum(weights2, dim=1).unsqueeze(1) + 1e-4)
        weights3 = weights3 / (torch.sum(weights3, dim=1).unsqueeze(1) + 1e-4)
        weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)

        document_vector1 = torch.sum(sentence_outputs1 * weights1, dim=1)
        document_vector2 = torch.sum(sentence_outputs2 * weights2, dim=1)
        document_vector3 = torch.sum(sentence_outputs3 * weights3, dim=1)
        document_vector4 = torch.sum(sentence_outputs4 * weights4, dim=1)



        # bs*(8*gru_size)
        document_vector=torch.cat((document_vector1, document_vector2, document_vector3,document_vector4), 1)
        #bs * class_num

        output4 = self.ec11(self.dropout(self.relu(self.bn4(self.ec1(document_vector1)))))
        output5 = self.ec21(self.dropout(self.relu(self.bn5(self.ec2(document_vector2)))))
        output6 = self.ec31(self.dropout(self.relu(self.bn6(self.ec3(document_vector3)))))
        output7=self.ec41(self.dropout(self.relu(self.bn7(self.ec4(document_vector4)))))

        output1 = self.fc11(self.dropout(self.relu(self.bn1(self.fc1(document_vector)))))
        output2 = self.fc21(self.dropout(self.relu(self.bn2(self.fc2(document_vector)))))
        output3 = self.fc31(self.dropout(self.relu(self.bn3(self.fc3(document_vector)))))

        return output1,output2,output3,output4,output5,output6,output7


if __name__ == "__main__":
    han_model = HAN_Model(vocab_size=100, embedding_size=200, gru_size=20, dropoutp=0.5,class_num1=4,class_num2=4,class_num3=4,element_num1=4,element_num2=4,element_num3=4,element_num4=4)
    x = torch.Tensor(np.zeros([64, 50, 100])).long()
    x[0][0][0:10] = 1
    output = han_model(x,x)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)