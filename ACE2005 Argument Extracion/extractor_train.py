import os
import sys
import random
import time
import datetime
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from data_loader import AceDataset
from extractor_model import ExtractorModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def set_seed(seed):
    random.seed(seed)  # random函数的随机种子
    np.random.seed(seed)  # np.random函数的随机种子
    torch.manual_seed(seed)  # pytorch的随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 固定卷积算法
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)  # pytorch.cuda的随机种子


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pad(batch):
    # 把batch里的token_id,label_id取出来，超过128的截取
    tokens_tensor = [s[0][0:128] for s in batch]
    labels_tensor = [s[1][0:128] for s in batch]

    # padding
    tokens_tensor = pad_sequence(tokens_tensor, batch_first=True)
    labels_tensor = pad_sequence(labels_tensor, batch_first=True)

    # mask矩阵
    masks_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long)
    masks_tensor = masks_tensor.masked_fill(tokens_tensor != 0, 1)

    return tokens_tensor, labels_tensor, masks_tensor


if __name__ == '__main__':
    # 确定种子
    set_seed(768)

    # 预训练模型路径
    model_name_or_path = '../bert-base-uncased/'
    tokenizer_name = 'vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_name_or_path, tokenizer_name),
                                              do_lower_case=True, cache_dir=None)

    # 一些初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 64

    # 读取数据
    train_set = AceDataset(tokenizer=tokenizer, dataset='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad)

    dev_set = AceDataset(tokenizer=tokenizer, dataset='dev')
    dev_loader = DataLoader(dev_set, batch_size=batch_size, collate_fn=pad)

    test_set = AceDataset(tokenizer=tokenizer, dataset='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pad)

    # 模型&损失函数&优化器
    role_size = train_set.num_roles
    model = ExtractorModel(batch_size=batch_size, role_total=role_size).to(device)

    alpha = 10
    weight = torch.zeros(role_size)
    weight[0] = 1
    weight[1:] = alpha
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model_params = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable Parameters Number: {len(model_params)}')

    optimizer = torch.optim.Adam(model_params, lr=1.0e-4)
    print(f'Current lr : {optimizer.param_groups[0]["lr"]}.')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # 训练轮数等训练相关参数
    decay_times = 0
    EPOCHES = 30
    patience = 0
    patience_threshold = 4
    best_f1 = 0.0
    best_loss = 99999.9

    # 保存文件
    f = open('log.txt', 'a')

    # 训练&验证
    for epoch in range(EPOCHES):
        print(f'--------------------------epoch {epoch + 1}--------------------------')
        # 统计单次 epoch 的训练时间
        t0 = time.time()
        # 重置每次epoch的训练总loss
        epoch_loss = 0.0
        # ----------------------------------------训练----------------------------------------
        model.train()

        for index, training_pair in tqdm(enumerate(train_loader)):
            # 取数据
            tokens_tensor, labels_tensor, masks_tensor = training_pair
            tokens_tensor = tokens_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            masks_tensor = masks_tensor.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward
            output = model(tokens_tensor, masks_tensor)

            # 计算loss
            dim1, dim2, _ = output.shape
            output = output.view(dim1 * dim2, -1)

            dim1, dim2 = labels_tensor.shape
            target = labels_tensor.view(dim1 * dim2)

            loss = criterion(output, target)

            # backward
            loss.backward()
            optimizer.step()

            # 累加到当前epoch的loss
            epoch_loss = epoch_loss + loss.item()
            # if index % 500 == 0:
            #     print(f'  Current Loss : {epoch_loss}')

        print(f'train loss: {epoch_loss:3f}')
        f.write(f'--------------------------epoch {epoch + 1}--------------------------\n')
        f.write(f'train loss: {epoch_loss:3f}\n')

        # ----------------------------------------验证----------------------------------------
        dev_loss = 0.0
        target_list = torch.zeros(1)
        pred_list = torch.zeros(1)

        model.eval()  # 预测模式
        with torch.no_grad():
            for index, training_pair in enumerate(dev_loader):
                # 取数据
                tokens_tensor, labels_tensor, masks_tensor = training_pair
                tokens_tensor = tokens_tensor.to(device)
                labels_tensor = labels_tensor.to(device)
                masks_tensor = masks_tensor.to(device)

                # forward
                output = model(tokens_tensor, masks_tensor)

                # 算loss
                dim1, dim2, _ = output.shape
                output = output.view(dim1 * dim2, -1)

                dim1, dim2 = labels_tensor.shape
                target = labels_tensor.view(dim1 * dim2)

                # 算loss,累加到当前epoch的loss
                loss = criterion(output, target)
                dev_loss = dev_loss + loss.item()

                # 计算准确率，召回率和F1
                _, prediction = torch.max(output, 1)

                if index == 0:
                    target_list = target
                    pred_list = prediction
                else:
                    target_list = torch.cat((target_list, target))
                    pred_list = torch.cat((pred_list, prediction))

        # 转到cpu算，否则报错
        target_copy = target_list.cpu()
        prediction_copy = pred_list.cpu()

        precision = precision_score(target_copy, prediction_copy, average='macro', zero_division=0)
        recall = recall_score(target_copy, prediction_copy, average='macro', zero_division=0)
        # f1 = f1_score(target_copy, prediction_copy, average='macro', zero_division=0)
        f1_v2 = 2 * precision * recall / (precision + recall)

        print(f'dev loss: {dev_loss:3f}\tprecision：{precision:3f}\trecall: {recall:3f}\tf1_score: {f1_v2:3f}')
        f.write(f'dev loss: {dev_loss:3f}\tprecision：{precision:3f}\trecall: {recall:3f}\tf1_score: {f1_v2:3f}\n')

        # ----------------------------------------测试----------------------------------------
        test_loss = 0.0
        target_list = torch.zeros(1)
        pred_list = torch.zeros(1)

        model.eval()  # 预测模式
        with torch.no_grad():
            for index, training_pair in enumerate(test_loader):
                # 取数据
                tokens_tensor, labels_tensor, masks_tensor = training_pair
                tokens_tensor = tokens_tensor.to(device)
                labels_tensor = labels_tensor.to(device)
                masks_tensor = masks_tensor.to(device)

                # forward
                output = model(tokens_tensor, masks_tensor)

                # 算loss
                dim1, dim2, _ = output.shape
                output = output.view(dim1 * dim2, -1)

                dim1, dim2 = labels_tensor.shape
                target = labels_tensor.view(dim1 * dim2)

                # 算loss,累加到当前epoch的loss
                loss = criterion(output, target)
                test_loss = test_loss + loss.item()

                # 计算准确率，召回率和F1
                _, prediction = torch.max(output, 1)

                if index == 0:
                    target_list = target
                    pred_list = prediction
                else:
                    target_list = torch.cat((target_list, target))
                    pred_list = torch.cat((pred_list, prediction))

        # 转到cpu算，否则报错
        target_copy = target_list.cpu()
        prediction_copy = pred_list.cpu()

        precision = precision_score(target_copy, prediction_copy, average='macro', zero_division=0)
        recall = recall_score(target_copy, prediction_copy, average='macro', zero_division=0)
        # f1 = f1_score(target_copy, prediction_copy, average='macro', zero_division=0)
        f1_v2 = 2 * precision * recall / (precision + recall)

        print(f'test loss: {test_loss:3f}\tprecision：{precision:3f}\trecall: {recall:3f}\tf1_score: {f1_v2:3f}')
        f.write(f'test loss: {test_loss:3f}\tprecision：{precision:3f}\trecall: {recall:3f}\tf1_score: {f1_v2:3f}\n')

        # ----------------------------------------根据testset保存----------------------------------------
        if best_f1 < f1_v2:
            best_f1 = f1_v2
            torch.save(model.state_dict(), 'model_save/best_f1_model.pth')

        if best_loss > test_loss:
            best_loss = test_loss
            # torch.save(model, 'model_save/best_loss.pkl')
            torch.save(model.state_dict(), 'model_save/best_loss_model.pth')
        else:
            patience = patience + 1

        # 计算当前epoch的用时
        t1 = time.time()
        training_time = format_time(t1 - t0)
        print("This epcoh took: {:}".format(training_time))

        if patience >= patience_threshold:
            if decay_times <= 3:
                scheduler.step()
                print(f'Current lr : {optimizer.param_groups[0]["lr"]}.')
                patience = 0
                decay_times = decay_times + 1
            else:
                continue

    # 收尾
    print('**************************************************')
    print(f'Best_loss(testset) : {best_loss:3f}\nBest_f1_score(testset) : {best_f1:3f}')
    print('**************************************************')

    f.write(f'\n\n\nBest_loss : {best_loss:3f}\nBest_f1_score : {best_f1:3f}\n')
