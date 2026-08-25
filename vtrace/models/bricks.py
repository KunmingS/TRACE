import copy
import math
import torch.nn.functional as F
import torch.nn as nn
from .builder import MODELS


@MODELS.register_module()
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,  # default to none to remind, act_cfg=dict(type="relu"),
    ):
        super().__init__()
        # norm config
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        # conv config
        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False  # bias is not necessary with a normalization layer

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:  # update conv_cfg_base
            conv_cfg_base.update(conv_cfg)

        # build conv layer
        self.conv = nn.Conv1d(**conv_cfg_base)

        # build norm layer
        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        # build activation layer
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)  # make a copy
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = nn.ReLU(inplace=True, **act_cfg)
            else:  # other type
                self.act = eval(act_type)(**act_cfg)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)

        if mask is not None:  # masking before the norm
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:  # masking the output
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]
            return x, mask
        else:
            return x


# ── merged from misc.py ──
import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# ── merged from transformer.py ──
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .builder import MODELS


@MODELS.register_module()
class TransformerBlock(nn.Module):
    """
    Adapted from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#L644

    Originally modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        in_channels,  # dimension of the input features
        n_head,  # number of attention heads
        n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # dimension of the hidden layer in MLP
        act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for the projection / MLP
        path_pdrop=0.0,  # drop path rate
        mha_win_size=-1,  # > 0 to use window mha
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        # layer norm for order (B C T)
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                in_channels,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
        else:
            self.attn = MaskedMHCA(
                in_channels,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * in_channels  # default
        if n_out is None:
            n_out = in_channels
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(in_channels, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float.unsqueeze(1) + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(
            self.mlp(self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)) * out_mask_float.unsqueeze(1)
        )
        return out, out_mask


@MODELS.register_module()
class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (replace position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsample feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # downsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, None, None, :]), float("-inf"))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, None, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        return out, qx_mask


@MODELS.register_module()
class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (replace position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsample feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        window_size,  # size of the local attention window
        n_qx_stride=1,  # downsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False,  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(x, (0, window_overlap + 1))
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(self, query, key, num_heads, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, num_heads, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(kv_mask[:, None, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap,
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        return out, qx_mask


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4, transpose=False):
        super().__init__()
        self.scale = nn.Parameter(init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True)
        self.drop_prob = drop_prob
        self.transpose = transpose  # if False, the input is B,C,T, otherwise, the input is B,T,C

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
            x = drop_path(self.scale * x, self.drop_prob, self.training)
            return x.transpose(1, 2)
        else:
            return drop_path(self.scale * x, self.drop_prob, self.training)

    def __repr__(self):
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"


# ── merged from sgp.py ──
import torch
import torch.nn.functional as F
import torch.nn as nn



class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=3,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        k=1.5,  # k
        group=1,  # group for cnn
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # hidden dim for mlp
        path_pdrop=0.0,  # drop path rate
        act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
        downsample_type="max",
        init_conv_vars=1,  # init gaussian variance for the weight
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = nn.LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == "max":
                kernel_size, stride, padding = n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == "avg":
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                    nn.Conv1d(n_embd, n_embd, 1, 1, 0),
                )
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        B, C, T = x.shape
        x = self.downsample(x)

        out_mask = F.interpolate(
            mask.unsqueeze(1).to(x.dtype),
            size=x.shape[-1],  # use actual downsampled feature size
            mode="nearest",
        ).detach()

        out = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.squeeze(1).bool()


# ── merged from gradient_ops.py ──
import torch
from torch.autograd import Function


class GradientScale(Function):
    """Scales gradients by a constant factor in the backward pass.

    Forward pass is identity. Backward pass multiplies gradient by `scale`.
    This allows partial gradient flow (e.g., scale=0.1) instead of full detach.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x, scale):
    """Apply gradient scaling to tensor x.

    Args:
        x: Input tensor.
        scale: Factor to multiply gradients by in backward pass.
            scale=0 is equivalent to detach(), scale=1 is identity.

    Returns:
        Tensor with same values but scaled gradients.
    """
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return GradientScale.apply(x, scale)
