""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline
label = [0, 0, 0, 0, 0, 0]
label_mask = [1, 1, 1, 1, 1, 1]


def rl_edu_to_sentence(edus, ind):
    ind = list(filter(lambda x: x < len(edus), ind))
    new_ind = []
    for i in ind:
        try:
            i = i.item()
        except:
            pass
        new_ind.append(i)
    ind = new_ind
    ind = list(sorted(ind))
    #print(ind)
    ret = ''
    for i in ind:
        ret += edus[i] + ' '
    rets = ret[:-1].split()
    masks = [1] * len(rets)
    return rets, masks


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch, sent_batch in loader:
            print(i)
            ext_sents = []
            ext_inds = []
            masks = []
            dirty = []
            for raw_arts, sent_labels in zip(art_batch, sent_batch):
                indices = agent(raw_arts, sent_labels)
                ext_inds += [(len(ext_sents), len(indices) - 1)]
                assert indices[-1][-1].item() == len(raw_arts) + 1
                tmp_stop = indices[-1][-1].item()
                tmp_truncate = tmp_stop - 1
                str_arts = list(map(lambda x: ' '.join(x), raw_arts))
                for idx in indices:
                    t, m = rl_edu_to_sentence(str_arts, idx)
                    if t == []:
                        assert len(idx) == 1
                        id = idx[0].item()
                        if id == tmp_truncate:
                            dirty.append(len(ext_sents))
                            ext_sents.append(label)
                            masks.append(label_mask)
                    else:
                        if idx[-1].item() != tmp_stop:
                            ext_sents.append(t)
                            masks.append(m)
            all_summs = abstractor(ext_sents, masks)
            for d in dirty:
                all_summs[d] = []
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
                if i % 100 == 1:
                    print(avg_reward/i, i)
                '''
                with open('./compare/rl/' + str(i - 1) + '.dec', 'w') as f:
                    for s in summs:
                        s = ' '.join(s)
                        f.write(s + '\n')
                '''
            #if i > 1000:
            #    break
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    masks = []
    art_batch, abs_batch, sent_label_batch = next(loader)
    leng = []
    avg_leng = []
    dirty = []
    time1 = time()
    for raw_arts, sent_labels in zip(art_batch, sent_label_batch):
        (inds, ms), bs = agent(raw_arts, sent_labels)
        assert inds[-1][-1].item() == len(raw_arts) + 1
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        try:
            avg_leng.append(sum(list(map(lambda x: len(x) - 1, inds[:-1])))/(len(inds) - 1))
        except:
            pass
        leng.append(len(inds) - 1)
        tmp_stop = inds[-1][-1].item()
        tmp_truncate = tmp_stop - 1
        str_arts = list(map(lambda x: ' '.join(x), raw_arts))
        for idx in inds:
            t, m = rl_edu_to_sentence(str_arts, idx)
            assert len(t) == len(m)
            if t == []:
                assert len(idx) == 1
                id = idx[0].item()
                if id == tmp_truncate:
                    dirty.append(len(ext_sents))
                    ext_sents.append(label)
                    masks.append(label_mask)
            else:
                if idx[-1].item() != tmp_stop:
                    ext_sents.append(t)
                    masks.append(m)
    print('长度：', leng)
    leng = list(map(lambda x: x[0] - len(x[1]), zip(leng, abs_batch)))
    avg_dis = sum(leng)/len(leng)
    print('平均差距：', avg_dis)
    print('平均edu数量：', sum(avg_leng)/len(avg_leng))
    time2 = time()
    with torch.no_grad():
        summaries = abstractor(ext_sents, masks)
        for d in dirty:
            summaries[d] = []
    time3 = time()
    i = 0
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch):
        rs_abs = []
        rs_len = []
        abs_num = min(len(inds) - 1, len(abss))
        for j in range(abs_num):
            rs_abs.append(reward_fn(summaries[i+j], abss[j]))
            rs_len.append(len(inds[j]))
        rs_zero = []
        for j in range(max(0, len(inds)-1-len(abss))):
            rs_zero += [0] * len(inds[j + abs_num])
        rs_zero += [0] * (len(inds[-1]) - 1)
        rs_final = stop_coeff*stop_reward_fn(list(concat(summaries[i:i+len(inds)-1])),
                                             list(concat(abss)))
        avg_reward += rs_final/stop_coeff
        i += len(inds)-1
        disc_rs = [rs_final]
        R = rs_final
        for _ in rs_zero:
            R = R * gamma
            disc_rs.append(R)
        for r, leng in zip(rs_abs, rs_len):
            R = r + R * gamma
            disc_rs += [R] * leng
        disc_rs = list(reversed(disc_rs))
        assert len(disc_rs) == sum(list(map(lambda x: len(x), inds)))
        rewards += disc_rs

    indices = list(concat(list(concat(indices))))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    assert len(reward) == len(probs) and len(baselines) == len(probs) and len(baselines) == len(indices)

    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))

    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-(p.log_prob(action))
                      * (advantage/len(indices))) # divide by T*B

    critic_loss = F.mse_loss(baseline, reward)
    time4 = time()
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    #print(avg_reward)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    time5 = time()
    print(time2-time1, time3-time2, time4-time3,time5-time4)
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff, model_path=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?
        if model_path is not None:
            self.load(model_path)
        self.param_debug()


    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)
        #return 0

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
