import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import random
from .rnn import MultiLayerLSTMCells
from .extract import PtrExtractSumm, LSTMPointerNet
INI = 1e-2

class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, extractor):
        super().__init__()
        self._sent_enc = extractor._sent_enc
        self._art_enc = extractor._art_enc
        ptr_net = extractor._extractor
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        self._truncate = nn.Parameter(extractor._truncate.clone())
        self._stop = nn.Parameter(extractor._stop.clone())

        #self.trans = nn.Parameter(extractor.trans)

        import numpy as np

        enc_out_dim = 512
        pre_pos = torch.tensor([10000**((i//2)/enc_out_dim) for i in range(enc_out_dim)])
        self.position = [[np.sin(p/pre_pos[i]) if i % 2 == 0 else np.cos(p/pre_pos[i])
                          for i in range(enc_out_dim)] for p in range(500)]



    def past_process(self, enc_out, sent_labels):
        def func(sent_label):
            return [self.position[p] for p in sent_label]
        x1 = list(map(func, sent_labels))
        position = torch.tensor(x1).to(enc_out.device)
        enc_out = torch.cat([enc_out, position], dim=2)
        enc_out = torch.matmul(enc_out, self.trans)
        return enc_out





    def forward(self, attn_mem, sent_labels):
        truncate = attn_mem.size(0)
        stop = attn_mem.size(0) + 1
        attn_mem = torch.cat([attn_mem,
                              self._truncate.unsqueeze(0),
                              self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        flag = False
        center = -1
        tmp_num = 0
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            #print(score.size())
            for o in outputs:
                if o != truncate:
                    score[0, o.item()] = -1e18


            if flag:
                #print(center)
                for point in range(truncate):
                    #print(sent_labels[point])
                    if abs(sent_labels[point] - center) > 1000:
                        score[0, point] = -1e18
            '''
            if flag:
                #print(center)
                for point in range(truncate):
                    #print(sent_labels[point])
                    if abs(sent_labels[point] - center) > 1000:
                        score[0, point] = -1e18
            '''
            #if tmp_num == 2:
            #    score[0, truncate] = 1e8
            #elif tmp_num < 2:
            #    score[0, truncate] = -1e8

            if len(outputs) > 100:
                score[0, truncate] = -1e18
                if len(outputs) > 110:
                    score[0, stop] = 1e8
                print(score)
                if self.training:
                    raise NotImplementedError
            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            tmp_num += 1
            if out.item() == truncate:
                flag = False
                tmp_num = 0
            elif not flag and out.item() != stop:
                center = sent_labels[out.item()]
                flag = True
            if out.item() == stop:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        new_extracts = []
        tmp = []
        #print(outputs)
        for e in outputs:
            tmp.append(e)
            if e == truncate or e == stop:
                new_extracts.append(tmp)
                tmp = []
        outputs = new_extracts
        if random.random() < 0.001:
            print(outputs)
        #print(outputs)
        #exit(0)
        if dists:
            # return distributions only when not empty (trining)
            return outputs, dists
        else:
            return outputs
    def decode(self, attn_mem, sent_labels):
        truncate = attn_mem.size(0)
        stop = attn_mem.size(0) + 1
        attn_mem = torch.cat([attn_mem,
                              self._truncate.unsqueeze(0),
                              self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        flag = False
        center = -1
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                 self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            for o in outputs:
                if o != truncate:
                    score[0, o.item()] = -1e18
            if flag:
                for point in range(truncate):
                    if abs(sent_labels[point] - center) > 1000:
                        score[0, point] = -1e18
            if len(outputs) > 100:
                score[0, truncate] = -1e18
                print(score)
            out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == truncate:
                flag = False
            elif not flag and out.item() != stop:
                center = sent_labels[out.item()]
                flag = True
            if out.item() == stop:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        new_extracts = []
        tmp = []
        for e in outputs:
            tmp.append(e)
            if e == truncate or e == stop:
                new_extracts.append(tmp)
                tmp = []
        outputs = new_extracts
        return outputs
    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output = torch.mm(score, attention)
        return output


class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)



    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(F.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, extractor, art_batcher):
        super().__init__()
        self._ext = PtrExtractorRL(extractor)
        self._scr = PtrScorer(extractor._extractor)
        self._batcher = art_batcher


    def forward(self, raw_article_sents, sent_labels, n_abs=None):
        #print(sent_labels)
        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._ext._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._ext._art_enc(enc_sent).squeeze(0)
        #print(enc_art.size())
        #enc_art = self._ext.past_process(enc_art.unsqueeze(0), [sent_labels]).squeeze()
        #print(enc_art.size())
        #exit(0)


        assert n_abs is None
        outputs = self._ext(enc_art, sent_labels)
        '''
        if self.training:
            outputs = self._ext(enc_art, sent_labels)
        else:
            print('!')
            outputs = self._ext.decode(enc_art, sent_labels)
        return outputs
        '''
        if self.training:
            #if n_abs is None:
            #    n_abs = len(outputs[0])
            n_abs = sum(list(map(lambda x: len(x), outputs[0])))
            scores = self._scr(enc_art, n_abs)
            return outputs, scores
        else:
            return outputs
       
