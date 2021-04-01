import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

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


class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        self.enc_bi_gru = torch.nn.GRU(input_size=768, hidden_size=256, dropout=0.2, bidirectional=True)
        self.detection_network_dense_out = torch.nn.Linear(512, 2)   # FCN
        self.soft_masking_coef_mapping = torch.nn.Linear(512, 1)  # FCN

    def forward(self, input_embeddings):

        h_0 = torch.zeros(2, input_embeddings.shape[1], 256)
        bi_gru_final_hidden_layer = self.enc_bi_gru(input_embeddings, h_0)[0]
        # 将隐藏层输出张量bi_gru_final_hidden_layer的第一第二维度互换,形状变为(batch_size, seq_len, enc_hid_size * 2)
        bi_gru_final_hidden_layer = bi_gru_final_hidden_layer.permute(1, 0, 2)

        detection_network_output = self.detection_network_dense_out(bi_gru_final_hidden_layer)  # 形状为(batch_size, seq_len, 2)

        soft_masking_coefs = torch.sigmoid(self.soft_masking_coef_mapping(bi_gru_final_hidden_layer))  # (batch_size, seq_len, 1)

        return detection_network_output, soft_masking_coefs


class CorrectionNetwork(BertPreTrainedModel):
    def __init__():
        pass


class SoftMaskedBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.config中包含了拼写错误纠正网络Correction_Network中的Bert模型的各种配置超参数.
        self.config = config

        self.detection_network = DetectionNetwork()
        # self.correct_network = CorrectionNetwork()

        '''二、构建的拼写错误纠正网络Correction_Network中BertModel中所用的个三种网络层'''

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

        self.mask_embeddings = self.embeddings.word_embeddings.weight[103]

        self.soft_masked_bert_dense_out = torch.nn.Linear(self.config.hidden_size,
                                                          self.embeddings.word_embeddings.weight.shape[0])

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def attention_coef(self, soft_masking_coefs, attention_mask):
        attention_mask = attention_mask.unsqueeze(dim=2)
        bool_attention_mask = (attention_mask != 0)   # 1 != 0 --> True   0 != 0 --> False
        # soft_masking_coefs[~bool_attention_mask] = 0   # False的coef=0
        hard_masking_coefs = torch.zeros_like(soft_masking_coefs)
        # print(bool_attention_mask.shape)
        batch_size = bool_attention_mask.size(0)
        sent_len = bool_attention_mask.size(1)
        for b in range(batch_size):
            for s in range(sent_len):
                if ~bool_attention_mask[b, s]:
                    hard_masking_coefs[b, s] = 0
                else:
                    hard_masking_coefs[b, s] = 1
        # print(soft_masking_coefs)
        return hard_masking_coefs


    def soft_mask(self, input_embeddings, soft_masking_coefs, attention_mask):
        soft_masking_coefs = self.attention_coef(soft_masking_coefs=soft_masking_coefs, attention_mask=attention_mask)

        self.repeated_mask_embeddings = self.mask_embeddings.unsqueeze(0).unsqueeze(0).repeat(1, input_embeddings.shape[1],
                                                                                     1).repeat(
            input_embeddings.shape[0], 1, 1)

        soft_masked_embeddings = soft_masking_coefs * self.repeated_mask_embeddings + (1 - soft_masking_coefs) * input_embeddings

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

        '''以下部分为transformers库中BertModel类中的forward()部门的一小部分源码, 放在此处是为了和源BertModel类保持一致防止出错.'''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # inputs_embeds.to(device)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)
        '''以上部分为transformers库中BertModel类中的forward()部门的一小部分源码, 放在此处是为了和源BertModel类保持一致防止出错.'''

        input_embeddings = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        # 形状变为(seq_len, batch_size, embed_size)->(seq_len, batch_size, 768).
        input_embeddings = input_embeddings.permute(1, 0, 2)

        detection_network_output, soft_masking_coefs = self.detection_network(input_embeddings)

        input_embeddings = input_embeddings.permute(1, 0, 2)
        soft_masked_embeddings = self.soft_mask(input_embeddings, soft_masking_coefs, attention_mask)

        '''拼写错误纠正网络Correction_Network'''
        '''soft_masked_embeddings输入错误纠正网络correction network的Bert模型后的结果经过最后的输出层与Softmax层后，
        即为句子中每个位置的字符经过错误纠正网络correction network计算后预测的正确字符索引结果的概率。'''

        '''注意: 最新版本的transformers.modeling_bert中的BertEncoder()类中forward()方法所需传入的参数中不再有output_attentions这个参数.'''
        encoder_outputs = self.encoder(soft_masked_embeddings,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        # outputs为一个包含四个元素的tuple：sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # outputs[0]代表Bert模型中最后一个隐藏层的输出(此时Bert模型中的隐藏层有12层,即num_hidden_layers参数为12),
        # 注意此处和循环神经网络的输出形状不同,循环网络隐藏层状态的输出为(seq_len, batch_size, bert_hidden_size)，
        # 此时outputs[0]的张量bert_output_final_hidden_layer的形状为(batch_size, seq_len, bert_hidden_size)—>(batch_size, seq_len, 768).
        bert_output_final_hidden_layer = outputs[0]

        # 注意!: 在soft_masked_embeddings输入拼写错误纠正网络correction network中的Bert模型后,其计算结果输入进最终的输出层与Softmax层之前，
        # 拼写错误纠正网络correction network的结果需通过残差连接residual connection与输入模型一开始的input embeddings相加，
        # 相加的结果才输入最终的输出层与Softmax层中做最终的正确字符预测。
        residual_connection_outputs = bert_output_final_hidden_layer + input_embeddings

        '''self.soft_masked_bert_dense_out即为拼写错误纠正网络correction network之后的输出层, 其会将经过残差连接模块residual connection之后
           的输出的维度由768投影到纠错词表的索引空间. (此处输出层self.soft_masked_bert_dense_out的输出final_outputs张量即可被视为Soft_Masked_BERT模型的最终输出).'''
        final_outputs = self.soft_masked_bert_dense_out(
            residual_connection_outputs)

        # 此处输出层self.soft_masked_bert_dense_out的输出final_outputs张量即可被视为Soft_Masked_BERT模型的最终输出.
        return detection_network_output, final_outputs


config = BertConfig.from_pretrained(
    "/home/wujinjie/pytorch_seq2seq/data/emb/chinese_L-12_H-768_A-12/bert_config.json")
soft_masked_bert = SoftMaskedBERT.from_pretrained(
    "/home/wujinjie/pytorch_seq2seq/data/emb/chinese_L-12_H-768_A-12/pytorch_model.bin", config=config)

if __name__ == '__main__':
    input_ids = torch.Tensor([[101, 768, 867, 117, 102, 0]]).long()

    attention_mask = torch.Tensor([[1, 1, 1, 1, 1, 0]]).long()  #
    token_type_ids = torch.Tensor([[0, 0, 0, 0, 0, 0]]).long()
    position_ids = torch.Tensor([[0, 1, 2, 3, 4, 5]]).long()
    print(input_ids.shape, attention_mask.shape)  # torch.Size([1, 6])
    detection_output, output = soft_masked_bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids)
    print(detection_output, output, detection_output.shape, output.shape)   # torch.Size([1, 6, 21128])
