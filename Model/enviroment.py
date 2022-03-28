# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


class ENV(nn.Module):
    def __init__(self, gru_size, dropoutp,class_num1, class_num2,class_num3,element_num1,element_num2,element_num3,element_num4):
        super(ENV, self).__init__()
        self.sentence_context1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context3 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        self.sentence_context4 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2 * gru_size, 1)), requires_grad=True)
        # self.to_context4=nn.Linear(2 * gru_size, 2 * gru_size)
        # multi-head
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
        # 要素类别
        self.ec1 = nn.Linear(2 * gru_size, 256)
        self.ec2 = nn.Linear(2 * gru_size, 256)
        self.ec3 = nn.Linear(2 * gru_size, 256)
        self.ec4 = nn.Linear(2 * gru_size, 256)
        self.ec11 = nn.Linear(256, element_num1)
        self.ec21 = nn.Linear(256, element_num2)
        self.ec31 = nn.Linear(256, element_num3)
        self.ec41 = nn.Linear(256, element_num4)
        # BN
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        # word level  sentence level转换 3*doc to 2*gru_size
        # self.keguan_word=nn.Linear(6 * gru_size, 2 * gru_size)
        # self.keguan_sentence = nn.Linear(6 * gru_size, 2 * gru_size)
        self.dropout = nn.Dropout(p=dropoutp)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, attention_sentence_outputs1,attention_sentence_outputs2,attention_sentence_outputs3,attention_sentence_outputs4,x_mask_word,w_mask1,w_mask2,w_mask3,w_mask4,sentence_outputs1,sentence_outputs2,sentence_outputs3,sentence_outputs4,keti,zhuti,zhuguan,keguan,law,gpu=False):
        # # 句子个数
        # sentence_num = x.shape[1]
        # # 句子长度
        # sentence_length = x.shape[2]
        # # x原来维度是bs*sentence_num*sentence_length，经过下面变化成二维的：(bs*sentence_num)*sentence_length
        # x = x.view([-1, sentence_length])
        # x_mask=x_mask.view([-1, sentence_length])
        # # 加了embedding维度：(bs*sentence_num)*sentence_length*embedding_size
        # x_embedding = self.embedding(x)
        # # word_outputs.shape:(bs*sentence_num)*sentence_length*(2*gru_size)，这里因为是双向gru所以是2*gru_size
        # x_embedding=self.dropout(x_embedding)
        # word_outputs1, word_hidden = self.word_gru1(x_embedding)
        # word_outputs2, word_hidden = self.word_gru2(x_embedding)
        # word_outputs3, word_hidden = self.word_gru3(x_embedding)
        # word_outputs4, word_hidden = self.word_gru4(x_embedding)
        # #
        # # word_outputs1=self.dropout(word_outputs1)
        # # # print(word_outputs1[0][0])
        # # word_outputs2 = self.dropout(word_outputs2)
        # # word_outputs3 = self.dropout(word_outputs3)
        # # word_outputs4 = self.dropout(word_outputs4)
        # #word_hidden正向和反向最后输出的结果
        # # attention_word_outputs.shape:(bs*sentence_num)*sentence_length*(2*gru_size)，这里对应原文公式5
        # attention_word_outputs1 = torch.tanh(self.word_dense1(word_outputs1))
        # attention_word_outputs2 = torch.tanh(self.word_dense2(word_outputs2))
        # attention_word_outputs3 = torch.tanh(self.word_dense3(word_outputs3))
        # attention_word_outputs4 = torch.tanh(self.word_dense4(word_outputs4))
        # # weights.shape:(bs*sentence_num)*sentence_length*1，，这里和下面一句对应原文公式6
        #
        # weights1 = torch.matmul(attention_word_outputs1, self.word_context1)
        # weights2 = torch.matmul(attention_word_outputs2, self.word_context2)
        # weights3 = torch.matmul(attention_word_outputs3, self.word_context3)
        # weights4 = torch.matmul(attention_word_outputs4, self.word_context4)
        #
        # # weights.shape: (bs * sentence_num) * sentence_length * 1
        # weights1 = F.softmax(weights1, dim=1)
        # weights2 = F.softmax(weights2, dim=1)
        # weights3 = F.softmax(weights3, dim=1)
        # weights4 = F.softmax(weights4, dim=1)
        # # print(weights.shape)
        # # 这里对x加一个维度：(bs*sentence_num)*sentence_length*1
        #
        # # x = x.unsqueeze(2)
        # x_mask_word=x_mask.unsqueeze(2)
        # # print('--')
        # # print(x.shape)
        # # print(weights.shape)
        # # print(x_mask_word.shape)
        # # 对x加的这个维度进行判断，如果这个维度上为0，表示这个地方是pad出来的，没有必要进行计算。
        # # 如果为1，则按weight进行计算比例
        # if gpu:
        #     weights1 = torch.where(x_mask_word != 0, weights1, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        #     weights2 = torch.where(x_mask_word != 0, weights2, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        #     weights3 = torch.where(x_mask_word != 0, weights3, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        #     weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        # else:
        #     weights1 = torch.where(x_mask_word != 0, weights1, torch.full_like(x_mask_word, 0, dtype=torch.float))
        #     weights2 = torch.where(x_mask_word != 0, weights2, torch.full_like(x_mask_word, 0, dtype=torch.float))
        #     weights3 = torch.where(x_mask_word != 0, weights3, torch.full_like(x_mask_word, 0, dtype=torch.float))
        #     weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float))
        #
        #
        # # 这里由于忽略掉了pad为0的部分，所以要按维度重新计算分布
        # weights1 = weights1 / (torch.sum(weights1, dim=1).unsqueeze(1) + 1e-4)
        # weights2 = weights2 / (torch.sum(weights2, dim=1).unsqueeze(1) + 1e-4)
        # weights3 = weights3 / (torch.sum(weights3, dim=1).unsqueeze(1) + 1e-4)
        # weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)
        # # bs*sentence_num*(2*gru_size)
        # sentence_vector1 = torch.sum(word_outputs1 * weights1, dim=1).view([-1, sentence_num, word_outputs1.shape[-1]])
        # sentence_vector2 = torch.sum(word_outputs2 * weights2, dim=1).view([-1, sentence_num, word_outputs2.shape[-1]])
        # sentence_vector3 = torch.sum(word_outputs3 * weights3, dim=1).view([-1, sentence_num, word_outputs3.shape[-1]])
        # sentence_vector4 = torch.sum(word_outputs4 * weights4, dim=1).view([-1, sentence_num, word_outputs4.shape[-1]])
        #
        # # sentence_outputs.shape:bs*sentence_num*(2*gru_size)
        # sentence_outputs1, sentence_hidden = self.sentence_gru1(sentence_vector1)
        # sentence_outputs2, sentence_hidden = self.sentence_gru2(sentence_vector2)
        # sentence_outputs3, sentence_hidden = self.sentence_gru3(sentence_vector3)
        # sentence_outputs4, sentence_hidden = self.sentence_gru4(sentence_vector4)
        # # 对应原文公式8：bs*sentence_num*(2*gru_size)
        # attention_sentence_outputs1 = torch.tanh(self.sentence_dense1(sentence_outputs1))
        # attention_sentence_outputs2 = torch.tanh(self.sentence_dense2(sentence_outputs2))
        # attention_sentence_outputs3 = torch.tanh(self.sentence_dense3(sentence_outputs3))
        # attention_sentence_outputs4  = torch.tanh(self.sentence_dense4(sentence_outputs4))
        # bs*sentence_num*1
        sentence_num=attention_sentence_outputs1.shape[1]

        weights1 = torch.matmul(attention_sentence_outputs1, self.sentence_context1)
        weights2 = torch.matmul(attention_sentence_outputs2, self.sentence_context2)
        weights3 = torch.matmul(attention_sentence_outputs3, self.sentence_context3)
        weights4 = torch.matmul(attention_sentence_outputs4, self.sentence_context4)

        # bs*sentence_num*1
        weights1 = F.softmax(weights1, dim=1)
        weights2 = F.softmax(weights2, dim=1)
        weights3 = F.softmax(weights3, dim=1)
        weights4 = F.softmax(weights4, dim=1)

        # _, indices1 = torch.topk(weights1, 4, dim=1)
        # _, indices2 = torch.topk(weights2,4, dim=1)
        # _, indices3 = torch.topk(weights3, 4, dim=1)
        # _, indices4 = torch.topk(weights4, 4, dim=1)
        #
        # w1_mask = torch.zeros_like(weights1)
        # w2_mask = torch.zeros_like(weights1)
        # w3_mask = torch.zeros_like(weights1)
        # w4_mask = torch.zeros_like(weights1)
        # if gpu:
        #     w1_mask.cuda()
        #     w2_mask.cuda()
        #     w3_mask.cuda()
        #     w4_mask.cuda()
        #
        # for i in range(weights1.shape[0]):
        #     w1_mask[i][indices1[i]] = 1
        #     w2_mask[i][indices2[i]] = 1
        #     w3_mask[i][indices3[i]] = 1
        #     w4_mask[i][indices4[i]] = 1


        # bs * sentence_num * sentence_length
        x_mask_sentence = x_mask_word.view(-1, sentence_num, x_mask_word.shape[1])
        # bs * sentence_num * 1
        x_mask_sentence = torch.sum(x_mask_sentence, dim=2).unsqueeze(2)

        x_mask_sentence1=x_mask_sentence.mul(w_mask1)
        x_mask_sentence2 =x_mask_sentence.mul(w_mask2)
        x_mask_sentence3 =x_mask_sentence.mul(w_mask3)
        x_mask_sentence4 =x_mask_sentence.mul(w_mask4)
        # num_mask1=w1_mask+w_mask1
        # num_mask2 = w2_mask + w_mask2
        # num_mask3 = w3_mask + w_mask3
        # num_mask4 = w4_mask + w_mask4
        #calculate reward
        # full1=torch.full_like(x_mask_sentence1, 0.5, dtype=torch.float)
        # # full_half1=torch.full_like(x_mask_sentence1, 0.5, dtype=torch.float)
        # full0 = torch.full_like(x_mask_sentence1, 0, dtype=torch.float)
        # if gpu:
        #     full1.cuda()
        #     full0.cuda()
        #
        # num2_reward_1 = torch.where(num_mask1 == 2, full1,
        #                        full0)
        # num2_reward_2 = torch.where(num_mask2 == 2, full1,
        #                        full0)
        # num2_reward_3 = torch.where(num_mask3 == 2, full1,
        #                        full0)
        # num2_reward_4 = torch.where(num_mask4 == 2, full1,
        #                        full0)
        #提供前期奖励
        # num0_reward_1 = torch.where(num_mask1 == 0, weights1,
        #                        full_half1)
        # num0_reward_2 = torch.where(num_mask2 == 0, weights1,
        #                        full_half1)
        # num0_reward_3 = torch.where(num_mask3 == 0, weights1,
        #                         full_half1)
        # num0_reward_4 = torch.where(num_mask4 == 0, weights1,
        #                        full_half1)
        # reward1=num2_reward_1+num0_reward_1
        # reward2 = num2_reward_2 + num0_reward_2
        # reward3 = num2_reward_3 + num0_reward_3
        # reward4 = num2_reward_4 + num0_reward_4
        # reward1 = num2_reward_1
        # reward2 = num2_reward_2
        # reward3 = num2_reward_3
        # reward4 = num2_reward_4

        if gpu:
            # bs * sentence_num * 1
            weights1 = torch.where(x_mask_sentence1 != 0, weights1, torch.full_like(x_mask_sentence1, 0, dtype=torch.float).cuda())
            weights2 = torch.where(x_mask_sentence2 != 0, weights2, torch.full_like(x_mask_sentence2, 0, dtype=torch.float).cuda())
            weights3 = torch.where(x_mask_sentence3 != 0, weights3, torch.full_like(x_mask_sentence3, 0, dtype=torch.float).cuda())
            weights4 = torch.where(x_mask_sentence4 != 0, weights4, torch.full_like(x_mask_sentence4, 0, dtype=torch.float).cuda())
        else:
            weights1 = torch.where(x_mask_sentence1 != 0, weights1, torch.full_like(x_mask_sentence1, 0, dtype=torch.float))
            weights2 = torch.where(x_mask_sentence2 != 0, weights2, torch.full_like(x_mask_sentence2, 0, dtype=torch.float))
            weights3 = torch.where(x_mask_sentence3 != 0, weights3, torch.full_like(x_mask_sentence3, 0, dtype=torch.float))
            weights4 = torch.where(x_mask_sentence4 != 0, weights4, torch.full_like(x_mask_sentence4, 0, dtype=torch.float))
        # bs * sentence_num * 1，对未计算的pad进行缩放
        weights1 = weights1 / (torch.sum(weights1, dim=1).unsqueeze(1) + 1e-4)
        weights2 = weights2 / (torch.sum(weights2, dim=1).unsqueeze(1) + 1e-4)
        weights3 = weights3 / (torch.sum(weights3, dim=1).unsqueeze(1) + 1e-4)
        weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)

        # bs *(2*gru)
        document_vector1 = torch.sum(sentence_outputs1 * weights1, dim=1)
        document_vector2 = torch.sum(sentence_outputs2 * weights2, dim=1)
        document_vector3 = torch.sum(sentence_outputs3 * weights3, dim=1)


        #客观要素重编码
        # word_outputs4, word_hidden = self.word_gru4(x_embedding)
        # word_context4=self.keguan_word(torch.cat((document_vector1, document_vector2, document_vector3), 1))
        # #  bs*(2*gru_size)
        # word_context4 = word_context4.unsqueeze(1).unsqueeze(3)
        # # bs*1*(2*gru_size)*1
        # word_context4 = word_context4.repeat(sentence_num, sentence_length, 1, 1)
        # # (bs*sentence_num)*sentence_length*(2*gru_size)*1
        # attention_word_outputs4 = torch.tanh(self.word_dense4(word_outputs4))
        # # (bs*sentence_num)*sentence_length*(2*gru_size)
        # attention_word_outputs4 = attention_word_outputs4.unsqueeze(3)
        # # (bs*sentence_num)*sentence_length*(2*gru_size)*1
        # attention_word_outputs4 = torch.transpose(attention_word_outputs4, 2, 3)
        # #(bs*sentence_num)*sentence_length*1*(2*gru_size)
        # weights4 = torch.matmul(attention_word_outputs4, word_context4)
        # # (bs*sentence_num)*sentence_length*1*1
        # weights4 = torch.squeeze(weights4, 3)
        # # (bs*sentence_num)*sentence_length*1
        # weights4 = F.softmax(weights4, dim=1)
        # if gpu:
        #     weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float).cuda())
        # else:
        #     weights4 = torch.where(x_mask_word != 0, weights4, torch.full_like(x_mask_word, 0, dtype=torch.float))
        # #
        # weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)
        # sentence_vector4 = torch.sum(word_outputs4 * weights4, dim=1).view([-1, sentence_num, word_outputs4.shape[-1]])
        #
        # #sentence level
        # sentence_outputs4, sentence_hidden = self.sentence_gru4(sentence_vector4)
        # attention_sentence_outputs4 = torch.tanh(self.sentence_dense4(sentence_outputs4))
        # sentence_context4 = self.keguan_sentence(torch.cat((document_vector1, document_vector2, document_vector3), 1))
        # #  bs*(6*gru_size)
        # sentence_context4 = sentence_context4.unsqueeze(1).unsqueeze(3)
        # # bs*1*(6*gru_size)*1
        # sentence_context4= sentence_context4.repeat(1, sentence_num, 1, 1)
        # # bs*sentence_num*(6*gru_size)*1
        # attention_sentence_outputs4 = torch.tanh(self.sentence_dense4(sentence_outputs4))
        # # bs*sentence_num*(2*gru_size)
        # attention_sentence_outputs4 = attention_sentence_outputs4.unsqueeze(3)
        # # bs*sentence_num*(6*gru_size)*1
        # attention_sentence_outputs4 = torch.transpose(attention_sentence_outputs4, 2, 3)
        # # bs*sentence_num*1*(6*gru_size)
        # weights4 = torch.matmul(attention_sentence_outputs4, sentence_context4)
        # # bs*sentence_num*1*1
        # weights4 = torch.squeeze(weights4, 3)
        # bs*sentence_num*1
        # weights4 = F.softmax(weights4, dim=1)
        # if gpu:
        #     # bs * sentence_num * 1
        #     weights4 = torch.where(x_mask_sentence != 0, weights4,
        #                            torch.full_like(x_mask_sentence, 0, dtype=torch.float).cuda())
        # else:
        #     weights4 = torch.where(x_mask_sentence != 0, weights4,
        #                            torch.full_like(x_mask_sentence, 0, dtype=torch.float))
        # # bs * sentence_num * 1，对未计算的pad进行缩放
        # weights4 = weights4 / (torch.sum(weights4, dim=1).unsqueeze(1) + 1e-4)
        document_vector4 = torch.sum(sentence_outputs4 * weights4, dim=1)



        # bs*(8*gru_size)
        document_vector=torch.cat((document_vector1, document_vector2, document_vector3,document_vector4), 1)
        #bs * class_num

        output4 = self.ec11(self.dropout(self.relu(self.bn4(self.ec1(document_vector1)))))
        output5 = self.ec21(self.dropout(self.relu(self.bn5(self.ec2(document_vector2)))))
        output6 = self.ec31(self.dropout(self.relu(self.bn6(self.ec3(document_vector3)))))
        output7 = self.ec41(self.dropout(self.relu(self.bn7(self.ec4(document_vector4)))))

        output1 = self.fc11(self.dropout(self.relu(self.bn1(self.fc1(document_vector)))))
        output2 = self.fc21(self.dropout(self.relu(self.bn2(self.fc2(document_vector)))))
        output3 = self.fc31(self.dropout(self.relu(self.bn3(self.fc3(document_vector)))))
        max1=torch.argmax(output4, 1)
        max1=keti-max1

        max2=torch.argmax(output5, 1)
        max2=zhuti-max2

        max3=torch.argmax(output6, 1)
        max3=zhuguan-max3

        max4=torch.argmax(output7, 1)
        max4=keguan-max4

        max_law = torch.argmax(output1, 1)
        max_law = max_law - law

        # max_accu = torch.argmax(output2, 1)
        # max_accu = max_accu - accu
        #
        # max_law = torch.argmax(output1, 1)
        # max_law = max_law - law
        #
        # max_term = torch.argmax(output3, 1)
        # max_term = max_term - term

        right_1 = torch.full_like(max1, 1, dtype=torch.float)
        wrong_1 = torch.full_like(max1, 0, dtype=torch.float)
        if gpu:
            right_1.cuda()
            wrong_1.cuda()
        right_reward_1 = torch.where(max1 == 0, right_1,
                                    wrong_1)
        right_reward_2 = torch.where(max2 == 0, right_1,
                                    wrong_1)
        right_reward_3 = torch.where(max3 == 0, right_1,
                                    wrong_1)
        right_reward_4 = torch.where(max4 == 0, right_1,
                                     wrong_1)
        right_reward_law = torch.where(max_law == 0, right_1,
                                     wrong_1)
        # right_reward_accu = torch.where(max_accu == 0, right_1,
        #                                 wrong_1)
        # right_reward_law = torch.where(max_law == 0, right_1,
        #                                wrong_1)
        # right_reward_term = torch.where(max_term == 0, right_1,
        #                                 wrong_1)
        right_reward_1 = right_reward_1+right_reward_law
        right_reward_2 = right_reward_2+right_reward_law
        right_reward_3 = right_reward_3+right_reward_law
        right_reward_4 = right_reward_4+right_reward_law

        right_reward_1=right_reward_1.reshape(right_reward_1.shape[0],-1).expand(right_reward_1.shape[0],15).unsqueeze(2)
        right_reward_2 = right_reward_2.reshape(right_reward_1.shape[0], -1).expand(right_reward_1.shape[0],15).unsqueeze(2)
        right_reward_3 = right_reward_3.reshape(right_reward_1.shape[0], -1).expand(right_reward_1.shape[0],15).unsqueeze(2)
        right_reward_4 = right_reward_4.reshape(right_reward_1.shape[0], -1).expand(right_reward_1.shape[0],15).unsqueeze(2)
        reward1=right_reward_1
        reward2=right_reward_2
        reward3=right_reward_3
        reward4=right_reward_4



        # print(output1.shape)
        return output1,output2,output3,output4,output5,output6,output7,reward1,reward2,reward3,reward4


if __name__ == "__main__":
    han_model = HAN_Model(vocab_size=100, embedding_size=200, gru_size=20, dropoutp=0.5,class_num1=4,class_num2=4,class_num3=4,element_num1=4,element_num2=4,element_num3=4)
    x = torch.Tensor(np.zeros([64, 50, 100])).long()
    x[0][0][0:10] = 1
    output = han_model(x,x)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)