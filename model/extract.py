import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize
INI = 1e-2


class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        # batch_size * encode_sent_num * hidden_size
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)  # batch_size * decode_sent_num * hidden_size
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            # batch_size * decode_sent_num * hidden_size
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        # batch_size * decode_sent_num * encode_sent_num
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, sent_labels):
        """extract k sentences, decode only, batch_size==1"""
        assert len(attn_mem) == 1 and len(mem_sizes) == 1
        sent_labels = sent_labels[0]
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        num = 0
        flag = False
        center = -1
        stop = mem_sizes[0] - 1
        truncate = mem_sizes[0] - 2
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()

            for e in extracts:
                if e < truncate:
                    score[e] = -1e6
            if flag:
                for point in range(truncate):
                    #print(sent_labels[point])
                    if abs(sent_labels[point] - center) > 1:
                        score[point] = -1e6
                #print(score)
            ext = score.max(dim=0)[1].item()
            if ext == truncate:
                flag = False
            elif not flag and ext != stop:
                #print(ext)
                #print(sent_labels)
                center = sent_labels[ext]
                flag = True
            extracts.append(ext)
            if ext == stop:
                break
            lstm_in = attn_mem[:, ext, :]
            lstm_states = (h, c)


        new_extracts = []
        tmp = []
        #print(extracts)
        for e in extracts:
            tmp.append(e)
            if e == truncate or e == stop:
                new_extracts.append(tmp)
                tmp = []
        extracts = new_extracts
        #print(extracts)
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output

import numpy as np
class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )
        self._truncate = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self._truncate, -INI, INI)
        self._stop = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self._stop, -INI, INI)

        #self.trans = nn.Parameter(torch.Tensor(enc_out_dim * 2, enc_out_dim))
        #pre_pos = torch.tensor([10000**((i//2)/enc_out_dim) for i in range(enc_out_dim)])
        #self.position = [[np.sin(p/pre_pos[i]) if i % 2 == 0 else np.cos(p/pre_pos[i])
        #                  for i in range(enc_out_dim)] for p in range(500)]


    def forward(self, article_sents, sent_labels, sent_nums, target):
        enc_out = self.encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)  # bs * num * dim
        #def func(sent_label):
        #    return [self.position[p] for p in sent_label]
        #x1 = list(map(func, sent_labels))
        #position = torch.tensor(x1).to(enc_out.device)
        #enc_out = torch.cat([enc_out, position], dim=2)
        #enc_out = torch.matmul(enc_out, self.trans)
        enc_out = torch.cat([enc_out, torch.zeros([bs, 2, d]).to(enc_out.device)], dim=1)
        for enc, num in zip(enc_out, sent_nums):
            enc[num] = self._truncate
            enc[num + 1] = self._stop
        sent_nums = list(map(lambda x: x + 2, sent_nums))
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, article_sents, sent_labels):
        assert len(article_sents) == 1 and len(sent_labels) == 1
        sent_nums = [len(article_sents[0])]
        enc_out = self.encode(article_sents, sent_nums)
        d = enc_out.size(2)
        enc_out = torch.cat([enc_out, torch.zeros([1, 2, d]).to(enc_out.device)], dim=1)
        enc_out[0, sent_nums[0]] = self._truncate
        enc_out[0, sent_nums[0] + 1] = self._stop
        sent_nums[0] += 2
        output = self._extractor.extract(enc_out, sent_nums, sent_labels)
        return output

    def encode(self, article_sents, sent_nums):
        assert sent_nums is not None
        max_n = max(sent_nums)
        enc_sents = [self._sent_enc(art_sent)
                     for art_sent in article_sents]

        def zero(n, device):
            z = torch.zeros(n, self._art_enc.input_size).to(device)
            return z

        enc_sent = torch.stack(
            [torch.cat([s, zero(max_n-n, s.device)], dim=0)
               if n != max_n
             else s
             for s, n in zip(enc_sents, sent_nums)],
            dim=0
        )
        lstm_out = self._art_enc(enc_sent, sent_nums) #bs * sent_num * hidden
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
