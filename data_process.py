import os
import json
import glob
import sys
articles = {}
total = 0
'''
edu切分的问题：
    跨越句子
    跨越单词（一个单词当做两个）
    RSB和LSB  " 和 ``
'''
import multiprocessing as mp


def find_diff(i):
    s = 'train'

    try:
        with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/'+s+'/'
                  + str(i) + '.json') as f:
            content = json.load(f)
            article = content['article']
        with open('/home/zhenwen/project/data/edu/dirty_'+s+'/' + str(i) + '.json') as f:
            content = json.load(f)
            edus = content['edu']
    except:
        return
    art1 = ''
    art2 = ''
    for art in article:
        art1 += art
    for art in edus:
        art2 += art
    art1 = art1.replace(' ', '')
    art2 = art2.replace(' ', '')
    art2 = art2.replace('-RSB-', ']')
    art2 = art2.replace('-LSB-', '[')
    
    if abs(len(art1) - len(art2)) > 30:
        print(i)
        #p = '/home/zhenwen/project/data/edu/'
        #os.popen('mv '+p+s+'/' + str(i) +'.json '+p+'dirty/'+s)
    return
    
    # art2 = art2.replace('``', '"')
    # art2 = art2.replace('``', '"')
    # print(abs(len(art1)- len(art2)))
    if len(art1) == len(art2):
        for j in range(len(art1)):
            if art1[j] != art2[j]:
                print('!!! ', i, j)
                break
    else:
        for j in range(min(len(art1), len(art2))):
            if art1[j] != art2[j]:
                print(art1[j - 10:j + 10], art2[j - 10:j + 10])
                break


my_char = '/'


def modify(i):
    split = 'test'
    try:
        try:
            with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/'+split+'/'
                      + str(i) + '.json') as f:
                content = json.load(f)
                article = content['article']
            with open('/home/zhenwen/project/data/edu/dirty_'+split+'/' + str(i) + '.json') as f:
                content = json.load(f)
                edus = content['edu']
        except:
            return
        art1 = ''
        art2 = ''
        dirty = 0
        for art in article:
            art1 += art.replace(' ', '')
        for art in edus:
            art2 += art.replace(' ', '')
            if art.replace(' ', '') == '':
                dirty += 1
                continue
            art2 += my_char
        art2 = art2.replace('-RSB-', ']')
        art2 = art2.replace('-LSB-', '[')
        print(len(art1), len(art2) - len(edus), abs(len(art1)-(len(art2) - len(edus))))
        p = 0
        pointer = []
        for j in range(len(art1)):
            if art1[j] != art2[j + p]:
                if art2[j + p] == my_char:
                    p += 1
                    pointer.append(j)
                    if j != len(art1) - 1 and art1[j+1] != art2[j+p+1] and art1[j] == '"':
                        p += 1
                elif art1[j] == '\"':
                    p += 1
                else:
                    print('not equal ', i)
                    #print(art1[j], art2[j+p])
                    #print(art1[j-10:j+10], art2[j + p -10:j + p+ 10])
                    #print('!!!!!!!')
                    #print('not equal!')
                    return
        #assert len(pointer) == len(edus) - dirty - 1
        pointer = list(map(lambda x: x - 1, pointer))
        point1 = []
        for k, art in enumerate(article):
            tmp = art.split()
            new_l = []
            for j, w in enumerate(tmp):
                w = w.replace(' ', '')
                new_l += [[k, j]] * len(w)
            assert len(new_l) == len(art.replace(' ', ''))
            point1 += new_l
        new_edus = []
        last = 0
        past_s = 0
        flag = False
        for l, p in enumerate(pointer):
            s, t = point1[p]
            if point1[p] == point1[pointer[l-1]]:
                continue
            tmp = article[s].split()
            tmps = ''

            if not flag and past_s != s:
                tmp1 = article[past_s].split()
                for j in range(last, len(tmp1)):
                    tmps += tmp1[j] + ' '
                last = 0
            for k in range(past_s + 1, s):
                tmp1 = article[k].split()
                for j in range(len(tmp1)):
                    tmps += tmp1[j] + ' '
            for j in range(last, t + 1):
                tmps += tmp[j] + ' '
            new_edus.append(tmps)
            last = t + 1
            if last >= len(tmp):
                flag = True
                last = 0
            else:
                flag = False
            past_s = s
        if pointer[-1] != len(art1) - 1:
            tmp = article[-1].split()
            tmps = ''
            if s != len(article) - 1 and not flag:
                tmp1 = article[s].split()
                for j in range(last, len(tmp1)):
                    tmps += tmp1[j] + ' '
                last = 0
            for k in range(s + 1, len(article) - 1):
                tmp1 = article[k].split()
                for j in range(len(tmp1)):
                    tmps += tmp1[j] + ' '

            for j in range(last, len(tmp)):
                tmps += tmp[j] + ' '
            new_edus.append(tmps)

        new_art = ''
        for x in new_edus:
            new_art += x.replace(' ', '')
        if new_art != art1:
            print("!!! ", i)
            return
        assert new_art == art1
        with open('/home/zhenwen/project/data/edu/dirty_'+split+'_new/' + str(i) + '.json', 'w') as f:
            content['edu'] = new_edus
            json.dump(content, f)
    except:
        print('报错 ', i)




def build_json(file):
    s = 'test'
    print(file)
    with open(file, 'r') as f:
        content = f.readlines()
        content = list(map(lambda x: x.replace('<S>', '').replace('\n', ''), content))
        content = {'edu': content}
    file_name = file[len('/home/zhenwen/data_edu_dirty_'+s+'/'):] + '.json'
    with open(
            '/home/zhenwen/project/data/cnn-dailymail/finished_files/'+s+'/' +
            file_name, 'r') as f:
        abs = json.load(f)['abstract']
        content.update({'abstract':abs})

    with open('/home/zhenwen/project/data/edu/dirty_'+s+'/' + file_name, 'w') as f:
        json.dump(content, f)
        #exit(0)


def make_sentence():
    import os
    import json
    import codecs
    path = '/home/zhenwen/project/data/cnn-dailymail/finished_files/test/'
    output = '/home/zhenwen/data_sentence_test/'

    fileList = list(sorted(os.listdir(path), key=lambda x: int(x[:-5])))
    for (i, file) in enumerate(fileList):
        #print(i)

        if i % 100 == 0:
            print(i)
            print(file)
        try:
            with open(path + file, 'r') as f:
                x = json.load(f)
        except:
            continue
        with codecs.open(output + file + '.article', 'w', 'utf-8') as f:
            for line in x['article']:
                f.write('<s> ' + line + ' </s>\n')


def multiproc(file_list, func):
    with mp.Pool() as pool:
        list(pool.imap_unordered(func, file_list, chunksize=1000))



def reparse(file_num):
    split = 'test'
    try:
        with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/' + split +
                  '/' + str(file_num) + '.json') as f:
            content = json.load(f)
            sent = content['article']
    except:
        return
    with open('/home/zhenwen/data_sentence_' + split + '/' + str(file_num), 'w') as f:
        for x in sent:
            f.write(x + '\n')

    


def compare():
    split = 'train'
    total_art = total_edu = total_art_sent = total_edu_sent = 0
    num = 0
    for i in range(290000):
        if i % 1000 == 0:
            print(i)
        try:
            with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/' + split + '/'
                      + str(i) + '.json') as f:
                content = json.load(f)
                article = content['article']
            with open('/home/zhenwen/project/data/edu/' + split + '/' + str(i) + '.json') as f:
                content = json.load(f)
                edus = content['edu']
        except:
            continue
        num += 1
        #print(len(article), len(edus))
        total_art += len(article)
        total_edu += len(edus)
        total_art_sent += sum([len(s) for s in article])
        total_edu_sent += sum([len(s) for s in edus])
    print(total_art, total_edu)
    print(total_art / num, total_edu / num)
    print('\n\n')
    print(total_art_sent, total_edu_sent)
    print(total_art_sent/total_art, total_edu_sent/total_edu)

def check_data():
    path = '/home/zhenwen/project/data/edu/train/'
    num = 0
    for i in range(290000):
        if i % 1000 == 0:
            print(i)
        file = path + str(i) + '.json'
        try:
            with open(file, 'r') as f:
                content = json.load(f)
        except:
            num += 1
            continue
        if 'edu' not in content.keys():
            print(i, 'not have edu')
        if 'abstract' not in content.keys():
            print(i, 'not have abstract')
        if 'extracted' not in content.keys():
            print(i, 'not have extracted')
        if 'score' not in content.keys():
            print(i, 'not have score')

        edu = content['edu']
        abstract = content['abstract']
        extracted = content['extracted']
        score = content['score']
        if len(edu) == 0:
            print(i, "length of edu is 0")
        if len(abstract) == 0:
            print(i, "length of abstract is 0")
        if len(extracted) == 0:
            print(i, "length of extracted is 0")
        if len(score) == 0:
            print(i, "length of score is 0")

        for e in extracted:
            if len(e) == 0:
                print(i, 'length of some ext is 0')
            if len(e) > 2:
                print(i, 'length of some ext is greater than 2')
        assert len(extracted) == len(score)
    print(num)


def find_null(i):
    path = '/home/zhenwen/project/data/cnn_another/'
    split = 'train'
    if not os.path.exists(path + split + '/' + str(i) + '.json'):
        print('不存在', i)
    else:
        try:
            with open(path + split + '/' + str(i) + '.json', 'r') as f:
                content = json.load(f)
        except:
            code = 'mv ' + path + split + '/' + str(i) + '.json' + ' ' + path + split + '_null/'
            print('内容为空', i)
            os.popen(code)


def delete_ext_blank(i):
    print(i)
    split = 'test'

    try:
        with open('/home/zhenwen/project/data/multi_sent_edu/' + split + '/' + str(i) + '.json') as f:
            content = json.load(f)
    except:
        return
    edu = content['edu']
    for e in edu:
        if e[-1] != ' ':
            print('!!!', i)
            return
    edu = list(map(lambda x: x[:-1], edu))
    content['edu'] = edu
    with open('/home/zhenwen/project/data/multi_sent_edu/' + split + '/' + str(i) + '.json', 'w') as f:
        json.dump(content, f,  indent=4, separators=(',', ':'))




def get_sentence_label(i):
    print(i)
    path1 = '/home/zhenwen/project/data/cnn-dailymail/finished_files/'
    path2 = '/home/zhenwen/project/data/sent_edu/'
    split = 'val'
    try:
        with open(path1 + split + '/' + str(i) + '.json') as f:
            content = json.load(f)
            sent = content['article']
        with open(path2 + split + '/' + str(i) + '.json') as f:
            content = json.load(f)
            edu = content['article']
    except:
        return
    s1 = ''
    s2 = ''
    for s in sent:
        s1 += s.replace(' ', '')
    for s in edu:
        s2 += s.replace(' ', '')
    assert s1 == s2
    p = 0
    edu_num = len(edu)
    flag = [0] * edu_num
    past = 0
    for num, s in enumerate(sent):
        sent_len = len(s.replace(' ', ''))
        edu_len = 0
        while p != edu_num and edu_len + past < sent_len:
            edu_len += len(edu[p].replace(' ', ''))
            flag[p] = num
            p += 1
        assert abs(edu_len + past - sent_len) < 25
        past = edu_len + past - sent_len
    assert p == edu_num
    with open(path2 + split + '/' + str(i) + '.json', 'w') as f:
        content['edu'] = content.pop('article')
        content['sentence'] = flag
        json.dump(content, f, indent=4, separators=(',', ':'))


#multiproc(glob.glob('/home/zhenwen/data_edu_test_dirty/*'), build_json)





#multiproc([i for i in range(290000)], find_diff)

#multiproc([i for i in range(290000)], modify)
#for i in range(290000):
#    modify(i)
#reparse()

#multiproc(list(glob.glob('/home/zhenwen/project/data/edu/dirty/test/*')), reparse)
#check_data()

#multiproc(list(range(287227)), find_null)
'''
split = 'val'
with open('/home/zhenwen/project/edu_summ/tmp_null_' + split, 'r') as f:
    content = list(map(lambda x: int(x), f.readlines()))
    #multiproc(content, reparse)
    for i in content:
        reparse(i)
'''
'''
136 val
'''
#multiproc([i for i in range(290000)], find_null)
#multiproc([i for i in range(19000)], delete_ext_blank)
#multiproc([i for i in range(19000)], get_sentence_label)
#for i in range(19000):
#    delete_ext_blank(i)


def check_data(i):
    print(i)
    pre = '/home/zhenwen/project/data/multi_sent_edu/'
    split = 'val'
    try:
        with open(pre + split + '/' + str(i) + '.json') as f:
            content = json.load(f)
    except:
        return
    ext = content['extracted']
    assert isinstance(ext, list)
    for e in ext:
        assert isinstance(e, list)

def count_abs():
    pre = './compare/rl/'
    total_len = total_num = 0
    for i in range(10000):
        try:
            with open(pre + str(i) + '.dec') as f:
                abs = f.readlines()
                total_len += len(abs)
                total_num += 1
        except:
            pass
    print(total_len)
    print(total_num)
    print(total_len/total_num)


def func():
    sent_num = 0
    word_num = 0
    abs_num = 0
    abs_word = 0
    # '/home/zhenwen/project/data/multi_sent_edu/train/'
    # '/home/zhenwen/project/data/cnn-dailymail/finished_files/train/'
    for i in range(290000):
        if i % 10000 == 0:
            print(i)
            print(word_num, sent_num)
        try:
            with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/train/' + str(i) + '.json') as f:
                content = json.load(f)
                article = content['edu']
                for sent in article:
                    if len(sent.split()) < 3:continue
                    sent_num += 1
                    word_num += len(sent.split())
                abst = content['abstract']
                for abs in abst:
                    abs_num += 1
                    abs_word += len(abs.split())
        except:
            pass

    print(sent_num)
    print(word_num)
    print(word_num/sent_num)

    print(abs_num)
    print(abs_word)
    print(abs_word/abs_num)


#func()
#count_abs()
#for i in range(19000):
#    check_data(i)

#multiproc([i for i in range(290000)], check_data)
from copy import deepcopy
def sent_to_edu(i):
    print(i)
    path='/home/zhenwen/project/data/cnn-dailymail/finished_files/test/'
    if not os.path.exists(path + str(i) + '.json'):return
    with open(path + str(i) + '.json') as f:
        try:
            content = json.load(f)
        except:
            return
        if content:
            if 'article' not in content.keys():return
            content['edu'] = deepcopy(content['article'])
            content.pop('article')
        else:
            return
    with open(path + str(i) + '.json', 'w') as f:
        json.dump(content, f, indent=4, separators=(',', ':'))


#multiproc([i for i in range(20000)], sent_to_edu)


def human_eval():
    l = [i for i in range(10000)]
    import random
    random.shuffle(l)
    l = l[:100]
    for point, i in enumerate(l):
        with open('./decode_dir/rl_all/output/' + str(i) + '.dec') as f:
            content = f.readlines()
        with open('/home/zhenwen/project/fast_abs_rl/decode_example/output/' + str(i) + '.dec') as f:
            baseline1 = f.readlines()
        with open('./decode_dir/ext_only/output/' + str(i) + '.dec') as f:
            baseline2 = f.readlines()
        with open('/home/zhenwen/project/data/multi_sent_edu/test/' + str(i) + '.json') as f:
            answer = json.load(f)['abstract']
        with open('/home/zhenwen/project/data/cnn-dailymail/finished_files/test/' + str(i) + '.json') as f:
            input = json.load(f)['edu']
        #print(content)
        #print(baseline1)
        #print(baseline2)
        #print(answer)
        #print(input)
        x = [content, baseline1, baseline2]
        y = [0, 1, 2]
        random.shuffle(y)

        with open('./human/output/' + str(point), 'w') as f:
            f.write(' '.join(input))
            f.write('\n\n\n')
            f.write(' '.join(answer))
            f.write('\n\n\n')
            for num in y:
                f.write(' '.join(x[num]))
                f.write('\n\n\n')
        with open('./human/label/' + str(point), 'w') as f:
            for num in y:
                f.write(str(num) + '  ')
            f.write('\n\n')
            f.write(str(i))
            f.write('\n')
        #exit(0)

#human_eval()

from rouge import Rouge
def my_compute(summ, abst, n=1, mode='f'):
    evaluator = Rouge(metrics=['rouge-n'],
                      max_n=2,
                      limit_length=True,
                      length_limit=200,
                      length_limit_type='words',
                      apply_avg='Avg',
                      apply_best='Best',
                      alpha=0.5,  # Default F1_score
                      weight_factor=1.2,
                      stemming=True)
    return evaluator.get_scores([summ], [abst])['rouge-' + str(n)][mode]


def compute_rouge():
    path1 = './decode_dir/large_only_ext/output/'
    path2 = '/home/zhenwen/project/data/multi_sent_edu/test/'
    avg_reward = 0
    tmp = 0
    leng = 0
    for i in range(3000, 11490):
        #if i == 136 or i == 3224:
        #    continue
        with open(path1 + str(i) + '.dec') as f:
            content = f.readlines()
        with open(path2 + str(i) + '.json') as f:
            ans = json.load(f)['abstract']

        summs = ' '.join(content)
        abs_sents = ' '.join(ans)
        #summs = summs.split()
        #abs_sents = abs_sents.split()
        #leng += len(summs)
        #avg_reward += compute_rouge_n(summs, abs_sents, n=1, mode='f')
        x = my_compute(summs, abs_sents, n=1, mode='f')
        tmp += x
        avg_reward += x
        #if i % 100 == 1:
            #print(avg_reward / i, i)
        if i % 100 == 99:
            print(tmp/100)
            tmp=0
    print('\n\n\n')
    print(avg_reward/(11490 - 3000))
    #print(leng)
    #print(leng/11490)
#compute_rouge()



def abstractive(n):
    from collections import Counter
    path1 = './decode_dir/large_only_ext/output/'
    path2 = '/home/zhenwen/project/data/multi_sent_edu/test/'

    def make_n_grams(seq, n):
        """ return iterator """
        ngrams = (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        return ngrams
    total = 0
    total_num = 0
    for i in range(11490):

        with open(path1 + str(i) + '.dec') as f:
            content = f.readlines()
        with open(path2 + str(i) + '.json') as f:
            doc = json.load(f)['edu']


        summs = ' '.join(content).split()
        doc = ' '.join(doc).split()

        c1 = set(make_n_grams(summs, n))
        #print(c1)
        #exit(0)
        c2 = set(make_n_grams(doc, n))
        '''
        for x in c1:
            if x not in c2:
                print(x)
        '''
        total += len(list(filter(lambda x: x not in c2, c1)))
        total_num += len(list(c1))
        if i % 100 == 0:
            print(total, total_num, total/total_num)
    print(total/total_num)
#abstractive(4)



