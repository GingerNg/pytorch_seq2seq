import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils.model_utils import use_cuda, device
import transformers
# 引入Bert的BertTokenizer与BertModel, 并单独取出BertModel中的词嵌入word_embeddings层
from transformers import BertConfig, BertModel, BertTokenizer
# 引入Bert模型的基础类BertEmbeddings, BertEncoder,BertPooler,BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel

'''
Soft_Masked_BERT模型,
Proposed in the Paper of ACL 2020: Spelling Error Correction with Soft-Masked BERT(2020_ACL)
基于https://github.com/wanglke/Soft-Masked-BERT.git的修改
'''


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        '''
        (1): 嵌入层BertEmbeddings(),其中包含了每个character的word embedding、segment embeddings、position embedding三种嵌入函数.
        (2): Bert模型的核心,多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.
        (3): Bert模型最后的池化层BertPooler.
        '''
        # 嵌入层BertEmbeddings().
        self.embeddings = BertEmbeddings(config)
        # 多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.   bert_encoder
        self.encoder = BertEncoder(config)
        # 池化层BertPooler。
        self.pooler = BertPooler(config)
        # 初始化权重矩阵,偏置等.
        self.init_weights()
        # self.mask_embeddings = self.embeddings.word_embeddings.weight[103]
        if use_cuda:
            self.to(device)
        #     self.mask_embeddings = self.mask_embeddings.to(device)  #

    def get_mask_embeddings(self, ind=103):
        """[获取某个字符的embedding]

        Args:
            ind (int, optional): [description]. Defaults to 103.

        Returns:
            [type]: [description]
        """
        return self.embeddings.word_embeddings.weight[ind].to(device)

    def encode(self, soft_masked_embeddings, extended_attention_mask):
        encoder_outputs = self.encoder(soft_masked_embeddings,
                                       attention_mask=extended_attention_mask,
                                       )
        sequence_output = encoder_outputs[0]
        # print(sequence_output.device)
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        # outputs为一个包含四个元素的tuple：sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # outputs[0]代表Bert模型中最后一个隐藏层的输出(此时Bert模型中的隐藏层有12层,即num_hidden_layers参数为12),
        # 注意此处和循环神经网络的输出形状不同,循环网络隐藏层状态的输出为(seq_len, batch_size, bert_hidden_size)，
        # 此时outputs[0]的张量bert_output_final_hidden_layer的形状为(batch_size, seq_len, bert_hidden_size)—>(batch_size, seq_len, 768).
        bert_output_final_hidden_layer = outputs[0]
        return bert_output_final_hidden_layer


class DetectionNetwork(nn.Module):
    def __init__(self, in_emb_dim=16):
        super(DetectionNetwork, self).__init__()
        self.enc_bi_gru = torch.nn.GRU(
            input_size=768, hidden_size=256, dropout=0.2, bidirectional=True)
        self.detection_network_dense_out = torch.nn.Linear(512, 2)   # FCN
        self.soft_masking_coef_mapping = torch.nn.Linear(512, 1)  # FCN
        if use_cuda:
            self.to(device)

    def forward(self, input_embeddings):
        # print(input_embeddings.shape)    # 50, 16, 768
        # h_0 = torch.zeros(2, input_embeddings.shape[1], 256)
        bi_gru_final_hidden_layer = self.enc_bi_gru(input_embeddings)[0]
        # 将隐藏层输出张量bi_gru_final_hidden_layer的第一第二维度互换,形状变为(batch_size, seq_len, enc_hid_size * 2)
        bi_gru_final_hidden_layer = bi_gru_final_hidden_layer.permute(1, 0, 2)

        detection_network_output = self.detection_network_dense_out(
            bi_gru_final_hidden_layer)  # 形状为(batch_size, seq_len, 2)

        soft_masking_coefs = torch.sigmoid(
            self.soft_masking_coef_mapping(bi_gru_final_hidden_layer)
        )  # (batch_size, seq_len, 1)

        return detection_network_output, soft_masking_coefs


class SoftMaskedBERT(nn.Module):
    def __init__(self, bert_path, config):
        super(SoftMaskedBERT, self).__init__()
        # self.config中包含了拼写错误纠正网络Correction_Network中的Bert模型的各种配置超参数.
        self.config = config
        self.bert_path = bert_path

        self.detection_network = DetectionNetwork()

        self.bert_model = BertModel.from_pretrained("{}/pytorch_model.bin".format(bert_path), config=config)

        self.soft_masked_bert_dense_out = torch.nn.Linear(self.config.hidden_size,
                                                          self.bert_model.embeddings.word_embeddings.weight.shape[0])
        self.all_parameters = {}
        # basic parameters
        parameters = []
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.detection_network.parameters())))
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.soft_masked_bert_dense_out.parameters())))
        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        # bert parameters
        # bert_parameters = self.get_bert_parameters()
        # self.all_parameters["bert_parameters"] = bert_parameters

        if use_cuda:
            self.to(device)

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def get_attention_coef(self, soft_masking_coefs, attention_mask):
        if self.training:  # model.train()
            attention_mask = attention_mask.unsqueeze(dim=2)
            # 1 != 0 --> True   0 != 0 --> False
            bool_attention_mask = (attention_mask != 0)
            # print(bool_attention_mask.device)
            # soft_masking_coefs[~bool_attention_mask] = 0   # False的coef=0

            # https://blog.csdn.net/ncc1995/article/details/99542594
            hard_masking_coefs = torch.zeros_like(soft_masking_coefs)
            if use_cuda:
                hard_masking_coefs = hard_masking_coefs.to(device)
            # print(bool_attention_mask.shape)
            batch_size = bool_attention_mask.size(0)
            sent_len = bool_attention_mask.size(1)
            for b in range(batch_size):
                for s in range(sent_len):
                    if ~bool_attention_mask[b, s]:  # 0 != 0 --> False
                        hard_masking_coefs[b, s] = 0
                    else:
                        hard_masking_coefs[b, s] = 1
            # print(soft_masking_coefs)
            return hard_masking_coefs
        else:  # # model.eval()
            return soft_masking_coefs

    def soft_mask(self, input_embeddings, soft_masking_coefs, attention_mask):
        soft_masking_coefs = self.get_attention_coef(soft_masking_coefs=soft_masking_coefs, attention_mask=attention_mask)

        # print(self.bert_model.get_mask_embeddings().device)
        repeated_mask_embeddings = self.bert_model.get_mask_embeddings().unsqueeze(0).unsqueeze(0).repeat(
                                                                                                    1, input_embeddings.shape[1], 1
                                                                                                ).repeat(
                                                                                                    input_embeddings.shape[0], 1, 1
                                                                                                )
        # print(self.bert_model.get_mask_embeddings().shape, repeated_mask_embeddings.shape)

        # print(self.bert_model.mask_embeddings.device, repeated_mask_embeddings.device, soft_masking_coefs.device)
        soft_masked_embeddings = soft_masking_coefs * repeated_mask_embeddings + (1 - soft_masking_coefs) * input_embeddings

        return soft_masked_embeddings

    '''forward函数.'''

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=None, ):

        # 利用张量的long()函数确保这些张量全为int型张量.
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        token_type_ids = token_type_ids.long()
        position_ids = position_ids.long()
        input_shape = input_ids.size()

        # [16,50] --> [16,1,1,50]
        extended_attention_mask = self.bert_model.get_extended_attention_mask(
            attention_mask, input_shape, device)
        # extended_attention_mask = extended_attention_mask.to(device)
        # print(extended_attention_mask.device)

        input_embeddings = self.bert_model.embeddings(input_ids=input_ids,
                                                      position_ids=position_ids,
                                                      token_type_ids=token_type_ids,
                                                      )
        # print("device input_embeddings:{}".format(input_embeddings.device))
        # 形状变为(seq_len, batch_size, embed_size)->(seq_len, batch_size, 768).
        permuted_input_embeddings = input_embeddings.permute(1, 0, 2)

        # DetectionNetwork
        detection_network_output, soft_masking_coefs = self.detection_network(permuted_input_embeddings)

        # soft_mask
        soft_masked_embeddings = self.soft_mask(input_embeddings, soft_masking_coefs, attention_mask)

        # print(soft_masked_embeddings.device)
        # bert_encode
        bert_output_final_hidden_layer = self.bert_model.encode(soft_masked_embeddings,
                                                                extended_attention_mask,
                                                                )
        # print("device bert_output_final_hidden_layer:{}".format(
        #     bert_output_final_hidden_layer.device))

        residual_connection_outputs = bert_output_final_hidden_layer + input_embeddings
        # print("device residual_connection_outputs:{}".format(
        #     residual_connection_outputs.device))

        '''self.soft_masked_bert_dense_out即为拼写错误纠正网络correction network之后的输出层, 其会将经过残差连接模块residual connection之后
           的输出的维度由768投影到纠错词表的索引空间. (此处输出层self.soft_masked_bert_dense_out的输出final_outputs张量即可被视为Soft_Masked_BERT模型的最终输出).'''
        final_outputs = self.soft_masked_bert_dense_out(
            residual_connection_outputs)

        # 此处输出层self.soft_masked_bert_dense_out的输出final_outputs张量即可被视为Soft_Masked_BERT模型的最终输出.
        return detection_network_output, final_outputs
