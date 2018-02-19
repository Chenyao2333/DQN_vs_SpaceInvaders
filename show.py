#! /usr/bin/env python3

import os


def parse_pair(p):
    name, value = p.split(":")
    name = name.strip()
    value = value.strip()
    
    try:
        value = int(value)
    except:
        pass

    return name, value

def read_scores(score_path):
    ret = []
    with open(score_path) as f:
        for l in f.readlines():
            it = {}
            ps = l.split(",")
            for p in ps:
                name, value = parse_pair(p)
                it[name] = value
            ret.append(it)
    return ret

def statistics():
    s = ""

    scores = read_scores("scores.txt")

    last_cnt = 0
    last_sum = 0
    for it in scores[-10:]:
        last_cnt += 1
        last_sum += it["Score"]

    best = 0
    for it in scores:
        if it["Score"] > best:
            best = it["Score"] 

    s += "Average last %d episode: %.2f\n" % (last_cnt, last_sum / last_cnt) 
    s += "Best score: %d\n" % best
    s += "Epsiodes: %d\n" % len(scores)
    
    return s

print(statistics())
#print(read_scores("scores.txt"))


