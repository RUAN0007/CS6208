# -*- coding: utf-8 -*-
# @Author: RUAN0007
# @Date:   2017-02-21 14:35:35
# @Last modified by:   RUAN0007
# @Last Modified time: 2017-02-22 10:02:04
#
#

import sys

import numpy as np
import json
import pickle as pk
from sklearn.manifold import TSNE
from icd9 import ICD9

def compress(embedding_path):
    raw_emb = np.load(embedding_path)
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    return model.fit_transform(raw_emb)

def load_code(code_path):
    code = pk.load(open(code_path,"rb"))
    code_i2s = code["code_i2s"]
    return code_i2s



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



def prepare_json(code_pkl, icd_json, emb_np):
    code_i2s = load_code(code_pkl)
    code_count = len(code_i2s)

    emb = compress(emb_np)

    med_emb = dict() # a dict where key is the medical code, value is a np matrix of embedding code

    tree = ICD9(icd_json)
    points = []
    for idx, code in code_i2s.iteritems():
        marker = {"symbol":"circle", "fillColor":get_color(code)}
        data = [list(emb[idx])]
        name = code + ": " + tree.find(code).description
        point = {"marker":marker, "data":data, "name":name}
        points.append(point)

    return json.dumps(points)


    # num_show = 10

    # for k,v in med_emb.iteritems():
    #     print "Med Code: ", k, " Embedding: ", v
    #     num_show -= 1
    #     if num_show == 0:
    #         break


if __name__ == "__main__":
    emb_path = "embedding_code_0.npy"
    icd_json = "icd9.json"
    code_pkl = "code.pkl"
    json_str = prepare_json(code_pkl, icd_json, emb_path)

    with open('emb.json', 'w') as f:
        f.write("emb='" + json_str + "'")  # python will convert \n to os.linesep

    # main()
