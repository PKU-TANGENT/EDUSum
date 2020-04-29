""" ROUGE utils"""
import os
import threading
import subprocess as sp
from collections import Counter, deque
from cytoolz import concat, curry
#from model.pyrouge import Rouge155
import re

from rouge import Rouge
r = Rouge()

def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count



@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    assert isinstance(output, list)
    evaluator = Rouge(metrics=['rouge-n'],
                      max_n=n,
                      limit_length=True,
                      length_limit=200,
                      length_limit_type='words',
                      apply_avg='Avg',
                      apply_best='Best',
                      alpha=0.5,  # Default F1_score
                      weight_factor=1.2,
                      stemming=True)

    output = ' '.join(output)
    reference = ' '.join(reference)
    score = evaluator.get_scores([output], [reference])['rouge-' + str(n)][mode]


    '''
    global_r = Rouge155()
    global_r.system_dir = './cal_rouge/summ'
    global_r.model_dir = './cal_rouge/ref'
    global_r.system_filename_pattern = '(\d+).dec'
    global_r.model_filename_pattern = '(\d+).ref'
    output = ' '.join(output)
    reference = ' '.join(reference)
    with open('./cal_rouge/ref/0.ref', 'w') as f:
        f.write(reference)
    with open('./cal_rouge/summ/0.dec', 'w') as f:
        f.write(output)
    output = global_r.convert_and_evaluate()
    if n == 1:
        y = re.search('ROUGE-1 Average_F: (.*?)\(', output)
    else:
        y = re.search('ROUGE-2 Average_F: (.*?)\(', output)
    score = float(y.group(0)[19:-2])
    '''
    #print(score)
    return score


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]



@curry
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    assert isinstance(output, list)
    evaluator = Rouge(metrics=['rouge-l'],
                      limit_length=True,
                      length_limit=200,
                      length_limit_type='words',
                      apply_avg='Avg',
                      apply_best='Best',
                      alpha=0.5,  # Default F1_score
                      weight_factor=1.2,
                      stemming=True)
    output = ' '.join(output)
    reference = ' '.join(reference)
    score = evaluator.get_scores([output], [reference])['rouge-l'][mode]
    '''
    global_r = Rouge155()
    global_r.system_dir = './cal_rouge/summ'
    global_r.model_dir = './cal_rouge/ref'
    global_r.system_filename_pattern = '(\d+).dec'
    global_r.model_filename_pattern = '(\d+).ref'
    output = ' '.join(output)
    reference = ' '.join(reference)
    with open('./cal_rouge/ref/0.ref', 'w') as f:
        f.write(reference)
    with open('./cal_rouge/summ/0.dec', 'w') as f:
        f.write(output)
    output = global_r.convert_and_evaluate()
    y = re.search('ROUGE-L Average_F: (.*?)\(', output)
    score = float(y.group(0)[19:-2])
    '''
    return score


def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None
class Meteor(object):
    def __init__(self):
        assert _METEOR_PATH is not None
        cmd = 'java -Xmx2G -jar {} - - -l en -norm -stdio'.format(_METEOR_PATH)
        self._meteor_proc = sp.Popen(
            cmd.split(),
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
            universal_newlines=True, encoding='utf-8', bufsize=1
        )
        self._lock = threading.Lock()

    def __call__(self, summ, ref):
        self._lock.acquire()
        score_line = 'SCORE ||| {} ||| {}\n'.format(
            ' '.join(ref), ' '.join(summ))
        self._meteor_proc.stdin.write(score_line)
        stats = self._meteor_proc.stdout.readline().strip()
        eval_line = 'EVAL ||| {}\n'.format(stats)
        self._meteor_proc.stdin.write(eval_line)
        score = float(self._meteor_proc.stdout.readline().strip())
        self._lock.release()
        return score

    def __del__(self):
        self._lock.acquire()
        self._meteor_proc.stdin.close()
        self._meteor_proc.kill()
        self._meteor_proc.wait()
        self._lock.release()
