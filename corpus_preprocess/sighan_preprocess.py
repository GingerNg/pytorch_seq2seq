import numpy as np
import logging
from nlp_tools.tokenizers import WhitespaceTokenizer
import torch
from utils.model_utils import use_cuda, device
from sklearn.model_selection import train_test_split
from utils import file_utils
import copy
import random

gCorrect = 0
gIncorrect = 1
G_VOCABULARY = [gCorrect, gIncorrect]


random_state = 42  # 固定随机种子

def read_data(set_type, year, ce):
    with open("/home/wujinjie/pytorch_seq2seq/sighan_raw/pair_data/simplified/{}{}_{}.txt".format(set_type, year, ce), "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def split_dataset(raw_path, train_path, dev_path, test_path):
    """[处理数据的起始点-分fold]

    Args:
        raw_path ([type]): [原始数据路径]
        train_path ([type]): [train-set存储路径]
        dev_path ([type]): [dev-set存储路径]
        test_path ([type]): [test-set存储路径]
    """
    # ce = ["correct", "error"]
    # set_type = ["train", "test"]
    # years = ["2015","2014","2013"]
    train_correct_2015 = read_data(set_type="train", year="15", ce="correct")
    train_correct_2014 = read_data(set_type="train", year="14", ce="correct")
    train_correct_2013 = read_data(set_type="train", year="13", ce="correct")
    train_corrects = train_correct_2015 + train_correct_2014 + train_correct_2013

    train_error_2015 = read_data(set_type="train", year="15", ce="error")
    train_error_2014 = read_data(set_type="train", year="14", ce="error")
    train_error_2013 = read_data(set_type="train", year="13", ce="error")
    train_errors = train_error_2015 + train_error_2014 + train_error_2013

    test_correct_2015 = read_data(set_type="test", year="15", ce="correct")
    test_correct_2014 = read_data(set_type="test", year="14", ce="correct")
    test_correct_2013 = read_data(set_type="test", year="13", ce="correct")
    test_corrects = test_correct_2015 + test_correct_2014 + test_correct_2013

    test_error_2015 = read_data(set_type="test", year="15", ce="error")
    test_error_2014 = read_data(set_type="test", year="14", ce="error")
    test_error_2013 = read_data(set_type="test", year="13", ce="error")
    test_errors = test_error_2015 + test_error_2014 + test_error_2013

    train_lines = []
    for i in range(len(train_corrects)):
        train_lines.append([train_corrects[i], train_errors[i]])

    test_lines = []
    for i in range(len(test_corrects)):
        test_lines.append([test_corrects[i], test_errors[i]])
    file_utils.write_bunch(train_path, train_lines)
    file_utils.write_bunch(dev_path, test_lines)
    file_utils.write_bunch(test_path, test_lines)


def process_dataset(config, train_path, dev_path):
    """[处理train-set，dev-set, 处理成模型合适的输入]

    Args:
        config ([type]): [description]
        train_path ([type]): [保存dev-set的路径]
        dev_path ([type]): [保存dev-set的路径]
    """
    def _process(path):
        """[summary]

        Args:
            path ([type]): [description]
        data_set [list[str], list[str],List[int]]
        Returns:
            [type]: [description]
        """
        dataset = []
        data_list = file_utils.read_bunch(path)
        for data in data_list:
            # print(data)
            g_labels = []
            left = list(data[0])[0:config["max_sequence_len"]-2]
            right = list(data[1])[0:config["max_sequence_len"]-2]
            for i in range(len(left)):
                if left[i] != right[i]:
                    g_labels.append(gIncorrect)
                else:
                    g_labels.append(gCorrect)
            dataset.append([left, right, g_labels])
        return dataset
    train_set = _process(train_path)
    dev_set = _process(dev_path)
    file_utils.write_json(config["train_set"], train_set)
    file_utils.write_json(config["dev_set"], dev_set)
    print("process_data　done")


class DatasetProcesser(object):
    """[语料的数值化]

    Args:
        object ([type]): [description]
    """
    def __init__(self, bert_path, config):
        super().__init__()
        self.max_sent_len = config["max_sequence_len"]
        self.tokenizer = WhitespaceTokenizer(bert_path, max_len=self.max_sent_len)

    def get_examples(self, data, label_encoder):
        examples = []
        for dat in data:
            # char --> char_id 数值化
            left_token_ids = self.tokenizer.tokenize(dat[0])
            right_token_ids = self.tokenizer.tokenize(dat[1])
            examples.append([left_token_ids, right_token_ids, dat[2], len(left_token_ids)])

        logging.info('Total %d docs.' % len(examples))
        return examples

    def data_iter(self, data, batch_size, shuffle=True, noise=1.0):
        """[生成batch]

        Args:
            data ([type]): [description]
            batch_size ([type]): [description]
            shuffle (bool, optional): [description]. Defaults to True.
            noise (float, optional): [description]. Defaults to 1.0.
        """
        def _batch_slice(data, batch_size):
            batch_num = int(np.ceil(len(data) / float(batch_size)))  # ceil 向上取整
            for i in range(batch_num):
                cur_batch_size = batch_size if i < batch_num - \
                    1 else len(data) - batch_size * i
                docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  # ???
                yield docs  # 　返回一个batch的数据

        batched_data = []
        if shuffle:
            np.random.shuffle(data)
            sorted_data = data
        else:
            sorted_data = data

        batch = list(_batch_slice(sorted_data, batch_size))
        batched_data.extend(batch)  # [[],[]]

        if shuffle:
            np.random.shuffle(batched_data)

        for batch in batched_data:
            yield batch

    def batch2tensor(self, batch_data):
        """batch2tensor"""
        batch_size = len(batch_data)
        doc_labels = []
        for doc_data in batch_data:
            if len(doc_data[2]) >= self.max_sent_len-2:
                doc_labels.append([gCorrect] + doc_data[2][0:self.max_sent_len-2] + [gCorrect])
            else:
                doc_labels.append([gCorrect] + doc_data[2] + [gCorrect]*(self.max_sent_len-len(doc_data[2])-1))
        batch_detection_labels = torch.LongTensor(doc_labels)

        token_type_ids = [0] * self.max_sent_len
        batch_inputs = torch.zeros(
            (batch_size, self.max_sent_len), dtype=torch.int64)
        batch_token_type_inputs = torch.zeros(
            (batch_size, self.max_sent_len), dtype=torch.int64)
        batch_correct_labels = torch.zeros(
            (batch_size, self.max_sent_len), dtype=torch.int64)
        batch_position_ids = torch.zeros(
            (batch_size, self.max_sent_len), dtype=torch.int64)

        for b in range(batch_size):
            token_ids = batch_data[b][0]
            correct_token_ids = batch_data[b][1]
            # ids = batch_data[b][0]
            for word_idx in range(min(self.max_sent_len, len(token_ids))):
                batch_inputs[b, word_idx] = token_ids[word_idx]
                batch_token_type_inputs[b, word_idx] = token_type_ids[word_idx]
                batch_correct_labels[b, word_idx] = correct_token_ids[word_idx]
                batch_position_ids[b, word_idx] = word_idx

        # if use_cuda:
        #     batch_inputs = batch_inputs.to(device)
        #     batch_token_type_inputs = batch_token_type_inputs.to(device)
        #     batch_detection_labels = batch_detection_labels.to(device)
        #     batch_correct_labels = batch_correct_labels.to(device)
        #     batch_position_ids = batch_position_ids.to(device)
        # print("batch_labels_shape:{}".format(batch_labels.shape))
        return batch_inputs, batch_token_type_inputs, batch_position_ids, batch_detection_labels, batch_correct_labels
