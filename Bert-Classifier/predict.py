import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertTokenizer
import os
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NewsDataset(Dataset):
    # 读取及初始化
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer

    # 取数据
    def __getitem__(self, idx):
        if self.mode == "test":
            query, news = self.df.iloc[idx, :2].to_numpy()
            label_tensor = None
        else:
            # 分成问题query、答案news和标签label，转numpy，再把label转tensor
            query, news, label = self.df.iloc[idx, :3].to_numpy()
            label_tensor = torch.tensor(label)

        # BERT分词&加入分隔符号[SEP]
        tokens_query = self.tokenizer.tokenize(query)
        word_pieces = ["[CLS]"] + tokens_query + ["[SEP]"]
        len_query = len(word_pieces)

        tokens_news = self.tokenizer.tokenize(news)
        word_pieces += tokens_news + ["[SEP]"]
        len_news = len(word_pieces) - len_query

        # token序列转换成索引序列，再转tensor
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 将第一句包含[SEP]的token位置置0, 第二句置1
        segments_tensor = torch.tensor([0] * len_query + [1] * len_news, dtype=torch.long)

        return tokens_tensor, segments_tensor, label_tensor

    def __len__(self):
        return self.len


# 这个函数的输入 `samples` 是一个 list，裡头的每个 element 都是
# 刚刚定义的 `NewsDataset` 回传的一个样本，每个样本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它会对前两个 tensors 作 zero padding，并產生的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 测试集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks，將 tokens_tensors 里 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    model.eval()  # 预测模式
    with torch.no_grad():
        for data in dataloader:
            # tensor丢到GPU
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda") for t in data if t is not None]

            # 把除了label的tensor传进去，前3个tensors分別为tokens,segments以及masks
            outputs = model(*data[:3])
            # 如果不传label，不返回loss
            logits = outputs[0]
            # output = torch.max(input, dim),dim=1每行最大值
            _, pred = torch.max(logits.data, 1)

            # 计算准确率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                score = f1_score(labels.cpu(), pred.cpu(), average='micro')

            # 记录预测的标签
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc, score
    else:
        return predictions


if __name__ == '__main__':
    # tokenizer为中文的bert-base-chinese
    NUM_LABELS = 3
    model = BertForSequenceClassification.from_pretrained('model_save', num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained('model_save')

    # 初始化Dataset类读数据，中文BERT断词
    testset = NewsDataset("test", tokenizer=tokenizer)

    # collate_fn为自定义如何去batch的数据，方法定义在create_mini_batch()函数
    BATCH_SIZE = 16
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    # 模型放到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # 训练前看看准确率
    predictions = get_predictions(model, testloader)
    print(type(predictions))
    print(len(predictions))

    predictions = predictions.numpy().tolist()


    with open('tmp.txt', 'w') as fp:
        fp.write('\n'.join('%s' % x for x in predictions))
