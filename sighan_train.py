from corpus_preprocess import sighan_preprocess as preprocess
import cfg
# from models.text_bert import BertSoftmaxModel
from models.smb import SoftMaskedBERT
from models.optimizers import Optimizer
import torch.nn as nn
import torch
import numpy as np
from utils import file_utils, model_utils, tensorboardX_utils
from evaluation_index import scores
import os
import time
from models import loss_factory
from utils.constants import MthStep
from transformers import BertConfig
import torch.optim as optim


config = cfg.config

dataset_name = "sighan"
model_name = "SoftMaskedBert"
writer = tensorboardX_utils.get_writer(
    "runs/{}/{}".format(dataset_name, model_name))

config.define("raw_path", "data/raw_data/%s" %
              "sighan", "path to raw dataset")  # 原始路径
config.define("save_path", "data/dataset/%s" %
              dataset_name, "path to save dataset")
config.define("glove_name", "840B", "glove embedding name")
# glove embedding path
# glove_path = '/data/dh/glove/glove.840B.300d.txt'
glove_path = './data/glove/glove.840B.300d.txt'
#glove_path = os.path.join(os.path.expanduser(''), "utilities", "embeddings", "glove.{}.{}d.txt")
config.define("glove_path", glove_path, "glove embedding path")
config.define("max_vocab_size", 50000, "maximal vocabulary size")

# 最大序列长度
config.define("max_sequence_len", 50,
              "maximal sequence length allowed")

config.define("min_word_count", 1, "minimal word count in word vocabulary")
config.define("min_char_count", 10,
              "minimal character count in char vocabulary")

# dataset for train, validate and test
config.define("vocab", "data/dataset/%s/pd_vocab.json" %
              dataset_name, "path to the word and tag vocabularies")

config.define("train_set", "data/dataset/%s/pd_train.json" %
              dataset_name, "path to the training datasets")
config.define("dev_set", "data/dataset/%s/pd_dev.json" %
              dataset_name, "path to the development datasets")

config.define("dev_text", "data/raw/LREC/2014_dev.txt",
              "path to the development text")

config.define("test_set", "data/dataset/demo/bert_ref.json",
              "path to the ref test datasets")
config.define("test_text", "data/raw/LREC/2014_test.txt",
              "path to the ref text")
config.define("pretrained_emb", "data/dataset/demo/glove_emb.npz",
              "pretrained embeddings")


config.define("cell_type", "lstm",
              "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
config.define("num_layers", 4, "number of rnn layers")
config.define("use_pretrained", False, "use pretrained word embedding")
config.define("tuning_emb", False,
              "tune pretrained word embedding while training")
config.define("emb_dim", 300,
              "embedding dimension for encoder and decoder input words/tokens")

config.define("train_batch_size", 16, "train_batch_size")
config.define("test_batch_size", 16, "test_batch_size")
config.define("epochs", 100, "epochs")
config.define("clip", 5.0, "clip")

# label_encoder = preprocess.LabelEncoer()
dataset_processer = preprocess.DatasetProcesser(cfg.bert_path, config=config)

raw_path = os.path.join(cfg.proj_path, config["raw_path"], "merge_syno.txt")

# tdt-set 存储路径
train_path = os.path.join(cfg.proj_path, config["raw_path"], "train.pl")
dev_path = os.path.join(cfg.proj_path, config["raw_path"], "dev.pl")
test_path = os.path.join(cfg.proj_path, config["raw_path"], "test.pl")


def run(mtd="fold_split"):
    if mtd in [MthStep.train.value]:
        # build the network model
        if not cfg.RESUME_EPOCH:
            bert_config = BertConfig.from_pretrained(
                "{}/bert_config.json".format(cfg.bert_path))
            # model = SoftMaskedBERT.from_pretrained(
            #     "{}/pytorch_model.bin".format(cfg.bert_path), config=bert_config)
            # model = SoftMaskedBERT(cfg.bert_path, label_encoder=None)
            model = SoftMaskedBERT(cfg.bert_path, bert_config)
        else:
            save_folder = os.path.join(cfg.proj_path, "data/models/{}/{}".format(dataset_name, model_name))
            print(' ******* Resume training from --  epoch {} *********'.format(cfg.RESUME_EPOCH))
            model = model_utils.load_checkpoint(os.path.join(
                save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))

    def _eval(data):
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        # print(model.training)
        y_pred = []
        y_true = []
        detection_y_pred = []
        detection_y_true = []
        with torch.no_grad():
            for batch_data in dataset_processer.data_iter(data, config['test_batch_size'], shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_token_type_inputs, batch_position_ids, batch_detection_labels, batch_correct_labels = dataset_processer.batch2tensor(
                    batch_data)
                detection_network_output, final_outputs = model(input_ids=batch_inputs,
                                                                attention_mask=batch_detection_labels,
                                                                token_type_ids=batch_token_type_inputs,
                                                                position_ids=batch_position_ids)
                # detection
                detection_output = detection_network_output.view(detection_network_output.size(0) * detection_network_output.size(1), detection_network_output.size(2))
                detection_y_pred.extend(torch.max(detection_output, dim=1)[1].cpu().numpy().tolist())
                detection_labels = batch_detection_labels.view(batch_detection_labels.size(0) * batch_detection_labels.size(1))
                detection_y_true.extend(detection_labels.cpu().numpy().tolist())

                # correct
                final_outputs = final_outputs.view(final_outputs.size(0) * final_outputs.size(1), final_outputs.size(2))
                # batch_outputs = final_outputs.detach().cpu().numpy().tolist()
                y_pred.extend(torch.max(final_outputs, dim=1)[1].cpu().numpy().tolist())
                correct_labels = batch_correct_labels.view(batch_correct_labels.size(0) * batch_correct_labels.size(1))
                y_true.extend(correct_labels.cpu().numpy().tolist())

            detection_score, detection_dev_f1 = scores.get_score(detection_y_true, detection_y_pred)
            score, dev_f1 = scores.get_score(y_true, y_pred)
        return score, dev_f1

    if mtd == MthStep.fold_split.value:
        preprocess.split_dataset(raw_path, train_path, dev_path, test_path)
    elif mtd == MthStep.process_data.value:
        preprocess.process_dataset(config, train_path, dev_path)
    elif mtd == MthStep.train.value:
        Train_data = file_utils.read_json(config["train_set"])
        Dev_data = file_utils.read_json(config["dev_set"])
        # 生成模型可处理的格式
        train_data = dataset_processer.get_examples(Train_data, label_encoder=None)
        dev_data = dataset_processer.get_examples(Dev_data, label_encoder=None)
        del Train_data, Dev_data
        # 一个epoch的batch个数
        batch_num = int(np.ceil(len(train_data) / float(config["train_batch_size"])))
        print("batch_num:{}".format(batch_num))
        # optimizer = Optimizer(model.parameters(),steps=batch_num * config["epochs"])  # 优化器
        # lr = 1e-3
        # betas = (0.5, 0.999)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=betas)
        optimizer = Optimizer(model.all_parameters, steps=batch_num * config["epochs"])  # 优化器
        best_train_f1, best_dev_f1 = 0, 0
        early_stop = -1
        EarlyStopEpochs = 5  # 当多个epoch，dev的指标都没有提升，则早停
        # train
        print("start train")
        for epoch in range(cfg.RESUME_EPOCH+1, config["epochs"] + 1):
            optimizer.zero_grad()
            model.train()  # 启用 BatchNormalization 和 Dropout
            overall_losses = 0
            step = 0
            for batch_data in dataset_processer.data_iter(train_data, config["train_batch_size"], shuffle=True):
                torch.cuda.empty_cache()
                batch_inputs, batch_token_type_inputs, batch_position_ids, batch_detection_labels, batch_correct_labels = dataset_processer.batch2tensor(
                    batch_data)
                # print(batch_inputs.shape)  # torch.Size([16, 100]) torch.Size([16, 100])
                detection_network_output, final_outputs = model(input_ids=batch_inputs,
                                                                attention_mask=batch_detection_labels,
                                                                token_type_ids=batch_token_type_inputs,
                                                                position_ids=batch_position_ids)
                # print(detection_network_output.shape, batch_detection_labels.shape)
                # print(final_outputs.shape, batch_correct_labels.shape)  # torch.Size([16, 100, 2]) torch.Size([16, 100, 21128])
                loss = loss_factory.smb_loss(detection_network_output, final_outputs, batch_detection_labels, batch_correct_labels)
                loss.backward(retain_graph=True)

                loss_value = loss.detach().cpu().item()  # 截断反向传播
                # losses += loss_value
                overall_losses += loss_value
                optimizer.step()
                step += 1
            overall_losses /= batch_num
            overall_losses = scores.reformat(overall_losses, 4)

            train_score, train_f1 = _eval(data=train_data)
            dev_score, dev_f1 = _eval(data=dev_data)

            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                early_stop = 0
                best_train_f1 = train_f1
                save_path = model_utils.save_checkpoint(
                    model, epoch, save_folder=os.path.join(cfg.proj_path, "data/models/{}/{}".format(dataset_name, model_name)))
                print("save_path:{}".format(save_path))
            else:
                early_stop += 1
                if early_stop == EarlyStopEpochs:  # 达到早停次数，则停止训练
                    print("train stopped")
                    break

            print("train_f1:{}, train_score[precision, recall, f1, acc]:{}, overall_loss:{} ".format(
                train_f1, train_score, overall_losses))
            print("dev_f1:{}, score:{}".format(dev_f1, dev_score))
            print("epoch:{},early_stop:{}, best_train_f1:{}, best_dev_f1:{}".format(
                epoch, early_stop, best_train_f1, best_dev_f1))
            print("-----------------")
            writer.add_scalars('score', {'train_score[precision, recall, f1, acc]': train_f1, "dev_score[precision, recall, f1, acc]": dev_f1,
                                         "best_train_f1": best_train_f1, "best_test_f1": best_dev_f1},
                               global_step=epoch)
            writer.add_scalars('loss', {"train_loss": overall_losses},
                               global_step=epoch)


if __name__ == "__main__":
    run(mtd=MthStep.fold_split.value)
    run(mtd=MthStep.process_data.value)
    run(mtd=MthStep.train.value)
