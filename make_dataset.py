#
"""Livedoorニュースコーパスから回帰用データセットを作成
"""

import os
import tarfile
import csv
import re
import requests
import random

from google_drive_downloader import GoogleDriveDownloader as gdd
from sudachipy import tokenizer
from sudachipy import dictionary

sdt = dictionary.Dictionary().create()

# ライブドアニュースコーパス
# https://www.rondhuit.com/download.html
# ライセンス: CC-BY-ND 2.1 https://creativecommons.org/licenses/by-nd/2.1/jp/
tgz_fname = "./ldcc-20140209.tar.gz"
# http://www.lr.pi.titech.ac.jp/~takamura/pndic_ja.html
# 単語感情極性対応表
# 高村大也, 乾孝司, 奥村学
# "スピンモデルによる単語の感情極性抽出", 情報処理学会論文誌ジャーナル, Vol.47 No.02 pp. 627--637, 2006.
pn_url = "http://www.lr.pi.titech.ac.jp/~takamura/pubs/pn_ja.dic"
pn_fname = os.path.basename(pn_url)

if not os.path.exists(tgz_fname):
    gdd.download_file_from_google_drive(file_id="1b-llzNQdmKIp0FYMwzGOKmXdQUNpNXC8",
                                    dest_path=tgz_fname, unzip=False)
if not os.path.exists(pn_fname):
    resp = requests.get(pn_url)
    with open(pn_fname, "w") as f:
        f.write(resp.content.decode('cp932'))
pn_words = {}
with open(pn_fname) as f:
    for line in f:
        row = line.strip().split(':')
        pn_words[row[0]] = float(row[-1])

def calc_sent_score(text):
    score = 0.0
    words = 0
    for word in sdt.tokenize(text):
        surface = word.surface()
        if surface in pn_words.keys():
            score += pn_words[surface]
            words += 1
    if words > 0:
        score /= words
    return score

target_genre = ["it-life-hack", "kaden-channel"]

zero_fnames = []
one_fnames = []
tsv_fname = "all.tsv"

brackets_tail = re.compile('【[^】]*】$')
brackets_head = re.compile('^【[^】]*】')

def remove_brackets(inp):
    output = re.sub(brackets_head, '',
                   re.sub(brackets_tail, '', inp))
    return output

def read_title(f):
    # 2行スキップ
    next(f)
    next(f)
    title = next(f) # 3行目を返す
    title = remove_brackets(title.decode('utf-8')).strip()
    return title

with tarfile.open(tgz_fname) as tf:
    # 対象ファイルの選定
    for ti in tf:
        # ライセンスファイルはスキップ
        if "LICENSE.txt" in ti.name:
            continue
        if target_genre[0] in ti.name and ti.name.endswith(".txt"):
            zero_fnames.append(ti.name)
            continue
        if target_genre[1] in ti.name and ti.name.endswith(".txt"):
            one_fnames.append(ti.name)
    with open(tsv_fname, "w") as wf:
        writer = csv.writer(wf, delimiter='\t')
        # ラベル 0
        for name in zero_fnames:
            f = tf.extractfile(name)
            title = read_title(f)
            score = calc_sent_score(title)
            row = [target_genre[0], 0, '', score, title]
            writer.writerow(row)
        # ラベル 1
        for name in one_fnames:
            f = tf.extractfile(name)
            title = read_title(f)
            score = calc_sent_score(title)
            row = [target_genre[1], 1, '', score, title]
            writer.writerow(row)

random.seed(100)
with open("all.tsv", 'r') as f, open("rand-all.tsv", "w") as wf:
    lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        wf.write(line)

random.seed(101)

train_fname, dev_fname, test_fname = ["train.tsv", "dev.tsv", "test.tsv"]

with open("rand-all.tsv") as f, open(train_fname, "w") as tf, open(dev_fname, "w") as df, open(test_fname, "w") as ef:
    ef.write("class\tsentence\n")
    for line in f:
        v = random.randint(0, 9)
        if v == 8:
            df.write(line)
        elif v == 9:
            row = line.split('\t')
            ef.write("\t".join([row[1], row[3]]))
        else:
            tf.write(line)
