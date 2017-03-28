# -*- coding: utf-8 -*-
# @Author: RUAN0007
# @Date:   2017-02-21 14:35:35
# @Last modified by:   RUAN0007
# @Last Modified time: 2017-02-23 12:59:19
#
#

import sys

import numpy as np
import json
import pickle as pk
from sklearn.manifold import TSNE
from icd9 import ICD9

import matplotlib.pyplot as plt

code_i2s = pk.load(open("data/code.pkl","rb"))["code_i2s"]
code_count = len(code_i2s)
tree = ICD9("data/icd9.json")

def compress(raw_emb):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    return model.fit_transform(raw_emb)


def get_color(code):
    if code[0] == "V":
        return "#000000" #Black
    if code[0] == "E":
        return "#696969" #Grey

    num_code = int(code)
    if num_code <= 139:
        return "#00FFFF"
    elif num_code <= 239:
        return "#0000FF"
    elif num_code <= 279:
        return "#7FFF00"
    elif num_code <= 289:
        return "#00008B"
    elif num_code <= 319:
        return "#9932CC"
    elif num_code <= 359:
        return "#FF1493"
    elif num_code <= 389:
        return "#1E90FF"
    elif num_code <= 459:
        return "#FFD700"
    elif num_code <= 519:
        return "#FF6984"
    elif num_code <= 579:
        return "#7CFC00"
    elif num_code <= 629:
        return "#F08080"
    elif num_code <= 679:
        return "#20B2AA"
    elif num_code <= 709:
        return "#00FF00"
    elif num_code <= 739:
        return "#800000"
    elif num_code <= 759:
        return "#000080"
    elif num_code <= 799:
        return "#FF4500"
    elif num_code <= 999:
        return "#663399"



def output_json(epoch, raw_emb, output_path):

    emb = compress(raw_emb)
    med_emb = dict() # a dict where key is the medical code, value is a np matrix of embedding code

    points = []
    for idx, code in code_i2s.iteritems():
        marker = {"symbol":"circle", "fillColor":get_color(code)}
        data = [list(emb[idx])]
        name = code + ": " + tree.find(code).description
        point = {"marker":marker, "data":data, "name":name}
        points.append(point)


    series = json.dumps(points).replace("'", "\\\'")
    with open(output_path, "w") as out:
        out.write("epoch = %d;\n" % epoch)
        out.write("emb = '%s';\n" % series)

def plot(train_stat, test_stat, title):
    '''Plotting train stat against epoch

    Args:
        train_stat: a list of train value indexed by epoch
        test_stat: a list of test value indexed by epoch
    '''
    plt.xlabel('Epoch')
    plt.title(title)
    train_line, = plt.plot(train_stat, marker='o', linestyle='-', color='r', label='Train')
    test_line, = plt.plot(test_stat, marker='x', linestyle='-', color='b', label='Test')
    plt.legend(handles=[train_line, test_line])
    plt.show(block=True)

def main():
    stat = pk.load(open("stat.pkl", 'rb'))
    # plot(stat["claim_train_recall"],
    #      stat["claim_test_recall"],
    #      "Visit-level Recall By Epoch")

    # stat["claim_train_recall"] = claim_train_recalls
    # stat["claim_test_recall"] = claim_test_recalls

    # stat["code_train_loss"] = code_train_losses
    # stat["code_test_loss"] = code_test_losses
    plt.subplot(121)
    plt.xlabel('Sub-epoch')
    plt.title("Visit-Level Loss")
    train_line, = plt.plot(stat['sub_claim_train_loss'][0:50], marker='o', linestyle='-', color='r', label='Train')
    test_line, = plt.plot(stat['sub_claim_test_loss'][0:50], marker='x', linestyle='-', color='b', label='Test')
    plt.legend(handles=[train_line, test_line])

    plt.subplot(122)
    plt.xlabel('Sub-epoch')
    plt.title("Visit-Level Recall")
    train_line, = plt.plot(stat['sub_claim_train_recall'][0:50], marker='o', linestyle='-', color='r', label='Train')
    test_line, = plt.plot(stat['sub_claim_train_recall'][0:50], marker='x', linestyle='-', color='b', label='Test')
    plt.legend(handles=[train_line, test_line])

    plt.show(block=True)

if __name__ == "__main__":
    main()
