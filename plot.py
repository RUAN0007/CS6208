# -*- coding: utf-8 -*-
# @Author: RUAN0007
# @Date:   2017-02-21 14:35:35
# @Last modified by:   RUAN0007
# @Last Modified time: 2017-02-21 15:28:45
#
#

import sys

import numpy as np
import pickle as pk
from sklearn.manifold import TSNE

def compress(embedding_path):
    raw_emb = np.load(embedding_path)
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    return model.fit_transform(raw_emb)

def load_code(code_path):
    code = pk.load(open("code.pkl","rb"))
    dcode_i2s = code["dcode_i2s"]
    pcode_i2s = code["pcode_i2s"]
    print pcode_i2s
    return dcode_i2s, pcode_i2s


def main():
    code_path = "code.pkl"
    dcode_i2s, pcode_i2s = load_code(code_path)
    dcode_len = len(dcode_i2s)

    emb_path = "embedding_code_0.npy"
    emb = compress(emb_path)

    med_emb = dict() # a dict where key is the medical code, value is a np matrix of embedding code

    for dgn_idx, dgn_code in dcode_i2s.iteritems():
        med_emb[dgn_code] = emb[dgn_idx]

    for prcd_idx, prcd_code in pcode_i2s.iteritems():
        med_emb[prcd_code] = emb[prcd_idx + dcode_len]

    print "Len Medical Embedding: ", len(med_emb)

    # num_show = 10

    # for k,v in med_emb.iteritems():
    #     print "Med Code: ", k, " Embedding: ", v
    #     num_show -= 1
    #     if num_show == 0:
    #         break


if __name__ == "__main__":
    main()
