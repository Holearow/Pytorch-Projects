import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_loader import AceDataset
from extractor_model import ExtractorModel
from sklearn.metrics import precision_score, recall_score, f1_score
from extractor_train import pad
from tqdm import tqdm


if __name__ == '__main__':
    # 预训练模型路径
    model_name_or_path = '../bert-base-uncased/'
    tokenizer_name = 'vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_name_or_path, tokenizer_name),
                                              do_lower_case=True, cache_dir=None)

    # 读数据
    batch_size = 1
    test_set = AceDataset(tokenizer=tokenizer, dataset='dev')
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pad)

    # 加载模型，保存加载模型教程 https://zhuanlan.zhihu.com/p/144487165
    model = ExtractorModel(batch_size=batch_size)
    model.load_state_dict(torch.load('model_save/best_f1_model.pth', map_location='cpu'))
    # model.load_state_dict(torch.load('model_save/best_f1_model.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    jieguo = []
    # 预测
    model.eval()
    with torch.no_grad():
        for index, training_pair in tqdm(enumerate(test_loader)):
            tokens_tensor, labels_tensor, masks_tensor = training_pair

            output = model(tokens_tensor, masks_tensor)

            dim1, dim2, _ = output.shape
            output = output.view(dim1 * dim2, -1)

            dim1, dim2 = labels_tensor.shape
            target = labels_tensor.view(dim1 * dim2)

            # 计算准确率，召回率和F1
            _, prediction = torch.max(output, 1)

            tokens_tensor = tokens_tensor.squeeze(0).tolist()
            target = target.tolist()

            prediction = prediction.tolist()

            # print(target)
            # print(prediction)
            # print(tokens_tensor)

            sentence = [tokenizer.convert_ids_to_tokens(n) for n in tokens_tensor]
            # print(' '.join(sentence))
            target_name = [test_set.idx_to_class[n] for n in target]
            prediction_name = [test_set.idx_to_class[n] for n in prediction]

            target = [str(i) for i in target]
            prediction = [str(i) for i in prediction]

            # print(target_name)
            # print(prediction_name)
            jieguo.append([' '.join(sentence), ' '.join(prediction), ' '.join(target), ' '.join(prediction_name),
                           ' '.join(target_name)])

    with open('devset_result.txt', 'w', encoding='utf-8') as f:
        f.write('*******************************************************\n')
        for n in jieguo:
            f.write('句子：' + n[0] + '\n')
            f.write('预测值：' + n[1] + '\n')
            f.write('真实值：' + n[2] + '\n')
            f.write('预测标签：' + n[3] + '\n')
            f.write('真实标签：' + n[4] + '\n')
            f.write('*******************************************************\n')