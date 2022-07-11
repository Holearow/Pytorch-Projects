import torch
import json
from torch.utils.data import Dataset


class AceDataset(Dataset):
    def __init__(self, path='../data/ACE05/', tokenizer=None, dataset='train'):
        self.path = path
        self.tokenizer = tokenizer
        self.dataset = dataset

        # 论元标签种类
        self.role_labels = ('None', 'Person', 'Place', 'Buyer', 'Seller', 'Beneficiary',
                            'Price', 'Artifact', 'Origin', 'Destination', 'Giver', 'Recipient',
                            'Money', 'Org', 'Agent', 'Victim', 'Instrument', 'Entity',
                            'Attacker', 'Target', 'Defendant', 'Adjudicator', 'Prosecutor', 'Plaintiff',
                            'Crime', 'Position', 'Sentence', 'Vehicle', 'Time-Within', 'Time-Starting',
                            'Time-Ending', 'Time-Before', 'Time-After', 'Time-Holds', 'Time-At-Beginning', 'Time-At-End')
        self.num_roles = len(self.role_labels)
        self.class_to_idx = dict(zip(self.role_labels, range(self.num_roles)))
        self.idx_to_class = dict(zip(range(self.num_roles), self.role_labels))

        # 读json文件
        file_name = self.path + self.dataset + '.json'
        with open(file_name, 'r') as f:
            data = json.load(f)

        # 每一句话和对应的标签
        self.sentences = []
        self.entity_argument_pairs = []

        for instance in data:
            # 先筛数据
            if instance['event_type'] == 'None':
                continue
            if len(instance['entities']) == 0:
                continue

            # token并起来得到一句话
            tokens = instance['tokens']

            sentence = " ".join(tokens)
            sentence = instance['event_type'] + ' ' + " ".join(instance['trigger_tokens']) + ' [SEP] ' + sentence
            self.sentences.append(sentence)

            entity_argument_pair = []
            entities = instance['entities']

            for entity in entities:
                try:
                    entity_tokens = entity['tokens']
                except:
                    entity_tokens = entity['token']

                entity_string = " ".join(entity_tokens)
                entity_role = entity['role']
                entity_argument_pair.append([entity_string, entity_role])

            self.entity_argument_pairs.append(entity_argument_pair)

        self.length = len(self.sentences)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print(self.entity_argument_pairs[index])
        text = self.sentences[index]
        marked_text = "[CLS] " + text + " [SEP]"

        tokenized_text = self.tokenizer.tokenize(marked_text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor(indexed_tokens)
        # tokens_list = indexed_tokens

        tokenized_sentence = " ".join(tokenized_text)
        tokenized_sentence_label = ['None' for _ in tokenized_text]
        assert len(tokenized_text) == len(tokenized_sentence_label), 'tokens和labels数量不一样！'

        for entity_item in self.entity_argument_pairs[index]:
            tokenized_entity = self.tokenizer.tokenize(entity_item[0])

            tokenized_entity_length = len(tokenized_entity)
            tokenized_entity_to_string = " ".join(tokenized_entity)

            start_index_in_string = tokenized_sentence.find(' ' + tokenized_entity_to_string + ' ') + 1

            start_part_in_string_to_list = tokenized_sentence[start_index_in_string:].split(' ')

            start_index = tokenized_text.index(start_part_in_string_to_list[0])
            end_index = start_index + tokenized_entity_length

            for i in range(start_index, end_index):
                if entity_item[1] == 'None':
                    continue
                if i == start_index:
                    label_content = entity_item[1]
                else:
                    label_content = entity_item[1]
                tokenized_sentence_label[i] = label_content

            # labels_list = [self.class_to_idx[n] for n in tokenized_sentence_label]
        labels_tensor = torch.tensor([self.class_to_idx[n] for n in tokenized_sentence_label])

        return tokens_tensor, labels_tensor


