#
"""Livedoorニュースコーパスから回帰用データセットを作成
"""

import os
import tarfile
import csv
import re
import requests

from google_drive_downloader import GoogleDriveDownloader as gdd

# ライブドアニュースコーパス
# https://www.rondhuit.com/download.html
# ライセンス: CC-BY-ND 2.1 https://creativecommons.org/licenses/by-nd/2.1/jp/
tgz_fname = "./ldcc-20140209.tar.gz"
# http://www.cl.ecei.tohoku.ac.jp/index.php?Open%20Resources/Japanese%20Sentiment%20Polarity%20Dictionary
# 日本語評価極性辞書（用言編）ver.1.0（2008年12月版）
# 著作者: 東北大学 乾・岡崎研究室 / Author(s): Inui-Okazaki Laboratory, Tohoku University
pn_url = "http://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/wago.121808.pn"
pn_fname = os.path.basename(pn_url)

if not os.path.exists(tgz_fname):
    gdd.download_file_from_google_drive(file_id="1b-llzNQdmKIp0FYMwzGOKmXdQUNpNXC8",
                                    dest_path=tgz_fname, unzip=False)
if not os.path.exists(pn_fname):
    resp = requests.get(pn_url)
    with open(pn_fname, "w") as f:
        f.write(resp.content)

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
            row = [target_genre[0], 0, '', title]
            writer.writerow(row)
        # ラベル 1
        for name in one_fnames:
            f = tf.extractfile(name)
            title = read_title(f)
            row = [target_genre[1], 1, '', title]
            writer.writerow(row)