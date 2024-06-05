import json
import os
import pickle
import logging

from character.name_match.tool.is_chinese import cleaning_name

dname_l_dict = {}


def set_log(log_dir, log_time):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d %H:%M:%S",
                        level=logging.INFO, filemode='w', filename=f'{log_dir}/{log_time}.log')


def load_json(*paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(*paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, *paths, ensure_ascii=False):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=4)


def save_pickle(data, *paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_author_index(name, dnames, l_must_in_r=False):
    '''获取 name 在 dnames 中的序号

    Args:
        name: 需要查找的名字
        dnames: 名字列表
        l_must_in_r: 名字列表中每个名字为全称时，name中的每个字符必须出现在匹配上的名字里

    Returns:
        当未找到时返回-1，否则返回其序号

    '''
    # l_must_in_r 当右边为全名时， 左边所有字符必须在右边出现
    # -----------
    name = name.lower()
    dnames = [n.replace('.', ' ').lower() for n in dnames]
    name_l = cleaning_name(name).split()

    hit_idx = []
    for aidx, dname in enumerate(dnames):
        if dname not in dname_l_dict.keys():
            dname_l_dict[dname] = cleaning_name(dname).split()
        dname_l = dname_l_dict[dname]
        first_char = [sp[0] for sp in dname_l]
        if l_must_in_r:
            is_ok = True
            for n in name_l:
                if not any(n in m for m in dname_l):
                    is_ok = False
                    break
            if not is_ok:
                continue
        if any(n in dname_l for n in name_l):  # 将名字中部分结构出现在待匹配名字上的序号加入列表
            hit_idx.append((aidx, dname_l, first_char, [n for n in name_l if n not in dname_l]))
    if len(hit_idx) == 1:
        return hit_idx[0][0]  # 当只有一个作者满足条件时，返回该作者序号
    new_hit_idx = []
    for aidx, dname_l, first_char, new_name_l in hit_idx:
        idxs = [dname_l.index(n) for n in name_l if n in dname_l]
        for i in idxs:
            first_char[i] = ''
        if any(n[0] in first_char for n in new_name_l):  # 匹配除完全匹配部分外，其余部分的首字母
            first_char = [fc for fc in first_char if fc != '']
            new_hit_idx.append((aidx, first_char, new_name_l))

    if len(new_hit_idx) == 1:
        return new_hit_idx[0][0]
    # 若上述方法无法找到匹配的名字，则通过比较不同名字的相似程度来决定最终返回的 index
    min_gap = 9999
    res_aidx = -1
    for aidx, first_char, new_name_l in new_hit_idx:
        gap = 0
        n_fc = [n[0] for n in new_name_l]
        for n in n_fc:
            if n not in first_char:
                gap += 1
        for n in first_char:
            if n not in n_fc:
                if n in ''.join(new_name_l):
                    gap += 0.9
                else:
                    gap += 1
        if gap < min_gap:
            min_gap = gap
            res_aidx = aidx
        elif gap == min_gap:
            res_aidx = -1

    hits = []
    if res_aidx == -1:
        for aidx, dname in enumerate(dnames):
            if not any(n not in dname for n in name_l):
                hits.append(aidx)
    if len(hits) == 1:
        return hits[0]
    return res_aidx


def main():
    pass


if __name__ == '__main__':
    main()
