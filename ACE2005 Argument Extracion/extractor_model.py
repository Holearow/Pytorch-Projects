import torch
from torch import nn
from transformers import BertModel


class Attention(nn.Module):
    def __init__(self, tag_size):
        super(Attention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(tag_size, tag_size)

    def forward(self, index, t_matrix):
        """
            1.矩阵转置的tensor.transpose()和tensor.permute()的教程：
                https://blog.csdn.net/xinjieyuan/article/details/105232802
            2.矩阵转置tensor.t()就可以！！！
            3.矩阵乘法的教程：https://www.codeleading.com/article/50715444326/
        """
        # 得到t_matrix各个维度的大小
        batch_size, seq_len, tag_size = t_matrix.shape

        # 得到t时刻个T_1（一共有seq_len个），维度为[batch_size, tag_size]
        t_index = t_matrix[:, index, :]
        t_index = self.linear(t_index)
        t_index = self.tanh(t_index)

        # 初始化z_matrix，依旧是计算α的用的那个z
        z_matrix = torch.zeros(batch_size, seq_len).to(self.device)

        for i in range(seq_len):
            # 把seq_len个T中的第i个取出来，维度为[batch_size, tag_size]
            t_i = t_matrix[:, i, :]

            # 两个按位乘，再求和，相当于每个batch行向量*列向量，得到[batch_size]的z_i
            tmp1 = t_index * t_i
            z_i = tmp1.sum(dim=1)

            # 把这个z_i存回之前初始化好的z_matrix在seq_len那个维度对应的第i列
            z_matrix[:, i, ] = z_i

        # z_matrix在seq_len的维度做softmax归一化
        a_matrix = self.softmax(z_matrix)

        # attention机制的那个求和算得分的公式
        t_att = torch.zeros(batch_size, tag_size).to(self.device)

        for i in range(seq_len):
            # 累加第i项
            tmp = a_matrix[:, i, ].reshape(-1, 1) * t_matrix[:, i, :]
            t_att = t_att + tmp

        return t_att


class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(TLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.t_lstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              batch_first=True,
                              dropout=dropout)

    def forward(self, inputs, h, c):
        # input维度[batch_size, 1, input_size(868)]
        # h和c的维度是[1, hidden_size(768)]
        outputs, (h_t, c_t) = self.t_lstm(inputs, (h, c))
        return outputs, c_t, h_t


class ExtractorModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, tag_size=100, batch_size=64, role_total=36):
        super(ExtractorModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.role_total = role_total

        self.bert_extractor = BertModel.from_pretrained('../bert-base-uncased/')
        '''for p in self.bert_extractor.parameters():
            p.requires_grad = False'''
        self.dropout = nn.Dropout(0.5)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tag_size = tag_size

        # TLSTM1和TLSTM2的初始化相关
        self.batch_size = batch_size

        self.lstm1 = TLSTM(self.hidden_size + self.tag_size, self.hidden_size)
        self.lstm2 = TLSTM(self.hidden_size + self.tag_size * 2, self.hidden_size)

        # TLSTM1输出的otput_i处理得到T1_i的线性层，TLSTM2同理
        self.linear1 = nn.Linear(self.hidden_size, self.tag_size)
        self.linear2 = nn.Linear(self.hidden_size, self.tag_size)

        # tag_att
        self.attention = Attention(self.tag_size)

        # 预测论元标签用
        self.linear3 = nn.Linear(self.tag_size, self.role_total)

        # 用于测试
        self.linear0 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_test = nn.Linear(self.hidden_size, self.role_total)

    def forward(self, tokens_tensor, masks_tensor):
        """
            第一步，得到BERT编码的词向量
        """
        # 用bert提取特征，hidden_state->[batch_size, sequence_length, word_embedding]
        bert_output = self.bert_extractor(input_ids=tokens_tensor, attention_mask=masks_tensor)
        bert_hidden_states = bert_output[0]
        # bert_hidden_states = self.dropout(bert_output[0])

        """
            第二步，上一步得到的词向量放到TLSTM_1里，得到每一步的T1_i
        """
        # bert提取的词向量初始化的tag权重拼在一起
        # bert_hidden_states = self.linear0(bert_hidden_states)
        bert_tensor_shape = bert_hidden_states.shape
        batch_size = bert_tensor_shape[0]
        seq_len = bert_tensor_shape[1]

        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        # 初始化0时刻的input_tags，维度为[batch_size, tag_size]；以及0时刻的h_0,c_0
        T_1s = torch.zeros(batch_size, seq_len, self.tag_size).to(self.device)
        input_tags = torch.zeros(batch_size, self.tag_size).to(self.device)
        hidden = h_0
        cell = c_0

        for i in range(seq_len):
            # 取batch里每个句子的第i个单词的词向量，维度为[batch_size, word_embedding]
            input_words = bert_hidden_states[:, i, :]

            # 拼接word_embedding和tag，dim=1得到batch*(word_len+tag_len)
            input_content = torch.cat((input_words, input_tags), dim=1)

            # 扩到3维让lstm接收，维度为[batch_size, 1, word_embedding+tag_len]
            input_content = input_content.unsqueeze(1)

            # 送到TLSTM1里，output为[batch_size, 1, hidden_size]，降回2维[batch_size, hidden_size]
            output, hidden, cell = self.lstm1(input_content, hidden, cell)
            output = output.squeeze(1)

            # output_i映射为T1_i，维度为[batch_size, tag_size]，用T1_i更新input_tags并保存
            input_tags = self.linear1(output)
            T_1s[:, i, :] = input_tags

        """
            第三步，词向量+T_Att+T2_i
        """
        # 初始化t0时刻的一些T_2i,hidden和cell目前存的就是TLSTM1最后时刻的东西，留着就行
        T_2s = torch.zeros(batch_size, seq_len, self.tag_size).to(self.device)
        input_tags = torch.zeros(batch_size, self.tag_size).to(self.device)

        for i in range(seq_len):
            # 取batch里每个句子的第i个单词的词向量，维度为[batch_size, word_embedding]
            input_words = bert_hidden_states[:, i, :]

            # 计算tag_att，维度为[batch_size, tag_size]
            input_att = self.attention(i, T_1s)

            # 拼接word_embedding,att和tag，dim=1得到batch*(word_len+tag_len*2)
            input_content = torch.cat((input_words, input_tags, input_att), dim=1)

            # 测试
            # input_content = torch.cat((input_words, input_tags), dim=1)

            # 扩到3维让lstm接收，维度为[batch_size, 1, word_embedding+tag_len*2]
            input_content = input_content.unsqueeze(1)

            # 送到TLSTM2里，output为[batch_size, 1, hidden_size]，降回2维[batch_size, hidden_size]
            output, hidden, cell = self.lstm2(input_content, hidden, cell)
            output = output.squeeze(1)

            # output_i映射为T2_i，维度为[batch_size, tag_size]，用T2_i更新input_tags并保存
            input_tags = self.linear2(output)
            T_2s[:, i, :] = input_tags

        """
            第四步，T_2s线性层改维度得到概率
        """
        # 映射到[batch_size, seq_len, role_total]
        role_predict = self.linear3(T_2s)
        return role_predict
