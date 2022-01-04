import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch.nn as nn
import time
import math
from scipy.optimize import curve_fit
import random
import os
import torch.nn.functional as F
import torch.autograd as autograd
import copy
from typing import Optional, Any , Tuple
import torch
from torch import Tensor
import functional as F2
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


x1,y1 = np.load("nh.npy")
x2 = x1
y2 = y1
x1_max = np.max(x1)
x1_min = np.min(x1)
y1_max = np.max(y1)
y1_min = np.min(y1)
x1 = (x1-x1_min) /(x1_max-x1_min)
y1 = (y1-y1_min) / (y1_max-y1_min)
x1 = x1 - x1[-1]
y1 = y1 - y1[-1]

for i in range(len(x1)-1):
    x2[i] = x1[i+1]
    y2[i] = y1[i+1]




torch.manual_seed(100)
np.random.seed(100)

input_window = 1  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size =150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



class MultiheadAttention1(torch.nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        # >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention1, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = torch.nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = torch.nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim,bias=False)
        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = torch.nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention1, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:


            return F2.multi_head_attention_forward1(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:

            return F2.multi_head_attention_forward1(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return torch.nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])



class Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F2.linear2(input, self.weight, self.bias),self.weight

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TransformerEncoderLayer1(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):
        super(TransformerEncoderLayer1, self).__init__()
        self.self_attn = MultiheadAttention1(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer1, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2,st_weight, value_v = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        return src2,value_v,st_weight

class TransformerEncoder1(torch.nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder1, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src.to(device)

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



class Tnn1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(50, 50, bias=False)
        self.encoder2 = nn.Linear(50, 50, bias=False)
        self.encoder3 = nn.Linear(50, 50, bias=False)

    def forward(self, src):
        output = self.encoder1(src)
        output = torch.relu(output)

        output = self.encoder2(output)
        output = torch.relu(output)

        output = self.encoder3(output)

        output = torch.tanh(output)

        return output

class Tnn2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(50, 50, bias=False)
        self.encoder2 = nn.Linear(50, 50, bias=False)
        self.encoder3 = nn.Linear(50, 50, bias=False)

    def forward(self, src):
        output = self.encoder1(src)
        output = torch.relu(output)

        output = self.encoder2(output)
        output = torch.relu(output)

        output = self.encoder3(output)

        output = torch.tanh(output)

        return output

class Tnn3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(50, 50, bias=False)
        self.encoder2 = nn.Linear(50, 50, bias=False)
        self.encoder3 = nn.Linear(50, 50, bias=False)

    def forward(self, src):
        output = self.encoder1(src)
        output = torch.relu(output)

        output = self.encoder2(output)
        output = torch.relu(output)

        output = self.encoder3(output)

        output = torch.tanh(output)

        return output

class Tnn4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(50, 50, bias=False)
        self.encoder2 = nn.Linear(50, 50, bias=False)
        self.encoder3 = nn.Linear(50, 50, bias=False)

    def forward(self, src):
        output = self.encoder1(src)
        output = torch.relu(output)

        output = self.encoder2(output)
        output = torch.relu(output)
        #
        output = self.encoder3(output)

        output = torch.tanh(output)

        return output


class Ende(nn.Module):
	def __init__(self):
		super().__init__()
		self.tnn1 = Tnn1()
		self.tnn2 = Tnn2()
		self.tnn3 = Tnn3()
		self.tnn4 = Tnn4()


	def forward(self,src_inputt,output55):
		a = src_inputt.shape[2]
		b = int(a / 2)

		src_input1 = src_inputt[:, :, 0:b]
		src_input2 = src_inputt[:, :, b:]
		# -------------------------------------------------------------------
		src_input1_1 = self.tnn1(src_input1)

		src_input1_2 = self.tnn2(src_input1)
		src_input1_2 = torch.exp(src_input1_2)
		src_input1_2 = torch.multiply(src_input1_2, src_input2)


		src_input1_new1 = src_input1
		src_input2_new1 = src_input1_2 + src_input1_1
		# -------------------------------------------------------------------

		src_input2_1 = self.tnn3(src_input2_new1)

		src_input2_2 = self.tnn4(src_input2_new1)
		src_input2_2 = torch.exp(src_input2_2)
		src_input2_2 = torch.multiply(src_input2_2, src_input1_new1)


		src_input1_new2 = src_input2_2 + src_input2_1
		src_input2_new2 = src_input2_new1

		src_inputt11 = torch.cat((src_input1_new2, src_input2_new2), 2)


		flg_x3 = output55[:, :, 0:b]
		flg_y3 = output55[:, :, b:]

		flg_y2 = flg_y3
		flg_x2 = torch.multiply((flg_x3 - self.tnn3(flg_y3)), torch.exp(-(self.tnn4(flg_y3))))
		flg_x1 = flg_x2
		flg_y1 = torch.multiply((flg_y2 - self.tnn1(flg_x2)), torch.exp(-(self.tnn2(flg_x2))))

		output555 = torch.cat((flg_x1, flg_y1), 2)
		return src_inputt11,output555




class TransAm(nn.Module):
    def __init__(self, feature_size=1, num_layers=1, dropout=0.):
        super(TransAm, self).__init__()
        self.bias = None
        self.src_mask = None

        self.encoder1 = nn.Linear(4, 100, bias=False)
        self.linear1 = nn.Linear(1, 1, bias=False)
        self.linear2 = nn.Linear(1, 1, bias=False)

        self.encoder_layer = TransformerEncoderLayer1(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_encoder = TransformerEncoder1(self.encoder_layer, num_layers=num_layers)

        self.ende = Ende()


    def forward(self, src):
        src_input = src
        src_inputt = self.encoder1(src_input)

        src_inputt11,_ = self.ende(src_inputt,src_inputt)

        src4_1 = src_inputt11.transpose(0, 2)

        output0, value_v0, real_v0 = self.transformer_encoder(src4_1)  #
        output0 = output0.transpose(0, 1)

        output0 = self.linear1(output0.transpose(0,2)).transpose(0,2)
        src_inputt111 = self.linear2(src_inputt11.transpose(0,2)).transpose(0,2)
        www1 = self.linear1.weight
        www2 = self.linear2.weight

        ls = value_v0 * www1
        lss = F.relu(-ls) / ls
        ls2 = www2
        lss2 = F.relu(-ls2) / ls2

        output4_trans = output0 + 2 * lss * output0

        output4_linear = src_inputt111 + 2 * lss2 * src_inputt111
        output4 =  output4_linear + output4_trans

        output55 = -output4+src4_1.transpose(0,2)

        _,output555 = self.ende(output55,output55)

        wg_sf = self.encoder1.weight

        wgg = torch.mm(wg_sf.T, wg_sf)
        wgg_inv = torch.inverse(wgg)
        wg_sf_inv = torch.mm(wg_sf,wgg_inv)
        output555 = torch.mm(output555.squeeze(0), wg_sf_inv).unsqueeze(0)
        look = output555

        haha = 0
        return look, haha



def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = input_data.shape[1]
    for i in range(L - tw):
        train_seq = input_data[:,i:i + tw]
        train_label = input_data[:,i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data():

    he1 = np.vstack((x1, y1))
    he2 = np.vstack((x2, y2))

    amplitude = np.vstack((he1, he2))
    dataset = amplitude

    sampels = 1
    train_data = dataset

    test_data = dataset[0:4*sampels,:]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[
                     :-output_window]  # todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack..

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?
    return train_sequence.to(device), test_data.to(device)


def get_batch(source, i, batch_size):

    if batch_size > len(source) - 1 - i:
        data = source[-batch_size:]
    else:
        data = source[i:i + batch_size]

    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 2))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 2))
    return input, target


def train(model,train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)


        dataa = np.squeeze(data[:, :, 0: 4, ], axis=3)
        targetsa = np.squeeze(targets[:, :, 0: 4, ], axis=3)

        optimizer.zero_grad()
        output,haha = model(dataa)

        loss = criterion(output, targetsa)+haha


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.6)

        optimizer.step()


        total_loss += loss.item()



        if  batch > 0:
            cur_loss = total_loss

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:12.12f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 ,
                cur_loss, cur_loss))


            start_time = time.time()


def plot_and_loss(eval_model ,data_source, epoch):
    eval_model.eval()

    total_loss = 0.

    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            dataa = np.squeeze(data, axis=3)

            targeta = np.squeeze(target, axis=3)
            output,wg_sf= eval_model(dataa)

            total_loss += criterion(output, targeta).item()#+max(0,value_v)+ max(0,real_v)

    return total_loss


# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps,epoch):
    eval_model.eval()

    data, _ = get_batch(data_source, 0, 1)

    data = data.transpose(1,3)

    data1a = data[:,:,:,0]

    data2a = data1a

    plot_b = data1a.cpu().detach().squeeze().numpy()


    with torch.no_grad():
        for i in range(0, steps):
            output,wg_sf = eval_model(data1a)
            data2a = torch.cat((data2a, output))

            data1a = output


    data1 = data2a.cpu().detach().squeeze().numpy()


    mmm1 = data1[:, 0]
    nnn1 = data1[:, 1]


    plt.figure()
    font = {"family": "serif",
            "serif": "Times New Roman",
            "weight": "normal"}
    plt.rc("font", **font)


    plt.plot(plot_b[0], plot_b[1], marker="x", color="blue", markersize=8,label="initial point")#,real_v
    plt.plot(x1, y1,".", color="black",label="demonstration trajectory")

    plt.plot(mmm1, nnn1, "--", color="purple",label="reproduction trajectory")

    plt.legend(loc='upper right')
    plt.show()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode

    total_loss = 0.
    eval_batch_size = 500
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            dataa = np.squeeze(data, axis=3)
            targetsa = np.squeeze(targets, axis=3)

            output,_ = eval_model(dataa)


            total_loss += criterion(output, targetsa).cpu().item()
    return total_loss







train_data, val_data = get_data()
model = TransAm().to(device)

criterion = nn.SmoothL1Loss()
lr = 0.01
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=lr,weight_decay =0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9999) 


epochs =50002  # The number of epochs
best_model = None


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model,train_data)
    x111 = random.uniform(-1, 1)
    y111 = random.uniform(-1, 1)

    a = torch.Tensor((x111,y111)).squeeze()


    if (epoch % 5000 == 0):

        val_loss = plot_and_loss(model, val_data, epoch)
        predict_future(model, val_data,len(x1),epoch)


        # if (epoch % 10000 == 0):
        #     torch.save(model.state_dict(),"models/epochs%s_parameter.pkl" % epoch)

    else:

        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (
                time.time() - epoch_start_time)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()

