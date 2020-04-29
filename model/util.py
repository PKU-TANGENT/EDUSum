import math

import torch
from torch.nn import functional as F


#################### general sequence helper #########################
def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean

import random
def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()
    tmp = random.random()
    if tmp < 0.01:
        record = [0.] * torch.sum(targets != pad_idx).item()
        trun_record = [0.] * torch.sum(targets != pad_idx).item()
        total = 0
        for t in targets:
            m = torch.max(t).item()
            for i, x in enumerate(t):
                if x.item() == m:
                    record[total + i] = 1.
                elif x.item() == m - 1:
                    trun_record[total + i] = 1.
            total += torch.sum(t != pad_idx).item()
        record = torch.tensor(record).to(targets.device)
        trun_record = torch.tensor(trun_record).to(targets.device)
    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    if tmp < 0.01:
        trunc = loss * trun_record
        stop = loss * record
        print('!!!!!!')
        print(sum(trunc)/sum(trun_record))
        print(sum(stop)/sum(record))
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


def multi_target_sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    '''
    loss = sequence_loss(logits, targets[::, ::, 0], xent_fn, pad_idx) +\
           sequence_loss(logits, targets[::, ::, 1], xent_fn, pad_idx)
    return loss
    '''

    bs, dec_num, _ = targets.size()
    sent_num = logits.shape[2]
    new_target = torch.zeros([bs, dec_num, sent_num]).cuda()
    weight = torch.ones([bs, dec_num, sent_num]).cuda()
    #logits:bs*dec_num*sent_num
    for i, b in enumerate(targets):
        for j, step in enumerate(b):
            if step[0] != pad_idx:
                new_target[i][j][step[0]] = 1
                new_target[i][j][step[1]] = 1
                weight[i][j][step[0]] = 10
                weight[i][j][step[1]] = 10
    # TODO:修改！！！！！
    mask = targets[::, ::, 0] != pad_idx
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous()
    new_target = new_target.masked_select(
        mask.unsqueeze(2).expand_as(new_target)
    ).contiguous()
    weight = weight.masked_select(
        mask.unsqueeze(2).expand_as(weight)
    )
    #if xent_fn:
    #    loss = xent_fn(logit, new_target, weight=weight)
    #else:
    loss = F.binary_cross_entropy_with_logits(logit, new_target, weight=weight, reduce=False)
    
    #for x, y, z in zip(logit, new_target, loss):
    #    if y.item() == 1:
    #        print(x.item(), torch.sigmoid(x).item(), y.item(), z.item())
    #    else:
    #        tmp = random.random()
    #        if tmp < 0.1:
    #            print(x.item(), torch.sigmoid(x).item(), y.item(), z.item())
    #exit(0)
    
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))

    return loss


def edu_to_sentence(edus, sent_labels, i, flag=False):
    if flag:
        past_i = i
        i = i[0]
    label = sent_labels[i]
    leng = len(sent_labels)
    i0 = i
    while i0 >= 0 and sent_labels[i0] == label: i0 -= 1
    i1 = i
    while i1 < leng and sent_labels[i1] == label: i1 += 1
    i0 += 1

    ret = ' '.join(edus[i0:i1])
    mask = [0] * len(ret.split(' '))
    assert len(ret.split(' ')) == sum([len(r.split(' ')) for r in edus[i0:i1]])
    leng = sum([len(r.split(' ')) for r in edus[i0:i]])
    for j in range(leng, leng + len(edus[i].split(' '))):
        mask[j] = 1
    if flag:
        i = past_i[1]
        leng = sum([len(r.split(' ')) for r in edus[i0:i]])
        for j in range(leng, leng + len(edus[i].split(' '))):
            mask[j] = 1
    return ret, mask
#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states
