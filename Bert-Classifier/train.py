import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertTokenizer
import datetime
import time
import os
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
    score = 0
    score_count = 0

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
                score += f1_score(labels.cpu(), pred.cpu(), average='micro')
                score_count += 1

            # 记录预测的标签
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc, score/score_count
    else:
        return predictions


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    # tokenizer为中文的bert-base-chinese
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 初始化Dataset类读数据，中文BERT断词
    trainset = NewsDataset("train", tokenizer=tokenizer)
    devset = NewsDataset("valid", tokenizer=tokenizer)

    # collate_fn为自定义如何去batch的数据，方法定义在create_mini_batch()函数
    BATCH_SIZE = 16
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=create_mini_batch)
    devloader = DataLoader(devset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    # 加载分类模型，bert-base-chinese，类别数量n_class=3
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 3
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    # 模型放到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # 训练前看看准确率
    _, acc, score = get_predictions(model, devloader, compute_acc=True)
    print("classification acc on devset(before tuned):", acc)
    print("classification micro-f1 on devset(before tuned):", score)

    output_dir = 'model_save/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model_params = get_learnable_params(model)

    # adam优化器更新参数
    optimizer = torch.optim.Adam(model_params, lr=1.0e-5)
    dev_acc = []
    best_score = 0
    EPOCHS = 10

    # 训练过程
    for epoch in range(EPOCHS):
        print(f'--------------------epoch {epoch + 1}--------------------')
        # 统计单次 epoch 的训练时间
        t0 = time.time()
        # 重置每次epoch的训练总loss
        running_loss = 0.0
        # 训练模式
        model.train()

        for data in trainloader:
            # 放到GPU
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            # 梯度清零
            optimizer.zero_grad()
            # forward
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)
            # 传进去的有label，outputs[0]为loss，outputs[1]为logits
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
            # 记录当前batch的loss
            running_loss += loss.item()

        # 单次 epoch 的训练时长
        t1 = time.time()
        training_time = format_time(t1 - t0)
        print("training epcoh took: {:}".format(training_time))
        print(f'train total loss: {running_loss:3f}')

        _, acc, score = get_predictions(model, devloader, compute_acc=True)
        dev_acc.append(acc)
        print(f'devset acc: {acc:3f}')
        print(f'devset micro-f1: {score:3f}')

        # 准确率好的话保存模型
        if score > best_score:
            best_score = score
            output_dir = 'model_save/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            break
