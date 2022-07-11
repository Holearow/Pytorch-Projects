# ACE 2005数据集论元抽取代码

##### 数据集输入输出示例

| 句子：[CLS] in ##jure wounded [SEP] the da ##vao medical center , a regional government hospital , recorded 19 deaths with 50 wounded [SEP] |
| ------------------------------------------------------------ |
| 预测值：0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 0 0       |
| 真实值：0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 0 0       |
| 预测标签：None None None None None None None None None None None None None None None None None None None None Victim None None |
| 真实标签：None None None None None None None None None None None None None None None None None None None None Victim None None |

##### 模型概述

1. BERT得到词向量
2. 送到魔改的T-LSTM-1编码（输入为词向量+Tag向量）
3. Self-Attention处理
4. 送到魔改的T-LSTM-2编码（输入为词向量+Tag向量+Tag-Att向量）

