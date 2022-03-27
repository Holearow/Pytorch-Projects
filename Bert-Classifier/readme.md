# 神州比赛初赛问题回答匹配代码

#### 金融问答数据集示例

| question                 | answer                                       | label     |
| ------------------------ | -------------------------------------------- | --------- |
| 工行贷款有放款的同学吗？ | 没有没有，没有。                             | 1(不确定) |
| 6950法郎等于多少人民币？ | 关键是要看能不能下款。                       | 0(不相关) |
| 赌博会冻结个人银行卡吗?  | 如果是招行账户，非正常用卡可能导致账户冻结。 | 2(相关)   |

#### 训练文件run.py用到的东西

- 继承torch.utils.data的Dataset类
- 调用sklearn.metrics计算F1值
- 调用并fine-tune预训练模型BERT——具体是BertForSequenceClassification
- 在torch.utils.data的Dataloader类的collate_fn参数传函数进一步处理数据得到batch
- torch.nn.utils.rnn中pad_sequence补齐sequence长度
- BERT参数attention_mask把padding给mask掉

#### 预测文件predict.py用到的东西

- torch.max(input, dim),dim=1每行最大值，两个返回值item和index

