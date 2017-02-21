# -*- coding: utf-8 -*-
# @Author: RUAN0007
# @Date:   2017-02-21 19:33:44
# @Last modified by:   RUAN0007
# @Last Modified time: 2017-02-21 20:19:37
#
#
from icd9 import ICD9
import cPickle as pickle
import collections

def preprocess_claim(code_pkl, claim_csv,output_pkl):

    code = pickle.load(open(code_pkl,"rb"))

    code_i2s = code["code_i2s"]
    code_s2i = code["code_s2i"]

    distinct_code_count = len(code_i2s)

    # patients = pickle.load(open("patient.pkl","rb"))

    claims = {}
    lc = 0
    with open(claim_csv) as infile:
        infile.next()
        pre_pid = ""
        patient_claim = collections.OrderedDict()
        try:
            claim_code_count = 0
            for line in infile:
                lc = lc + 1
                if (lc % 30000 == 0):
                    print "Process Line "+ str(lc)
                    print "Pid: " +pid + " Cid: " + cid

                info = line.split(",")
                pid = info[0]

                # if not pid in patients:
                #   print "At line " + str(lc)
                #   print pid + " not in patient demo records"

                if pid != pre_pid and lc!=1:
                    claims[pre_pid] = patient_claim
                    patient_claim = collections.OrderedDict()

                cid = info[1]
                claim_code = set()

                for idx in range(12,22):
                    code = info[idx].strip()[0:3]
                    if code == "":
                        continue

                    if code.isdigit():
                        code = code[0:3]
                    if code[0] == 'V':
                        code = code[0:3]
                    if code[0] == 'E'and len(code) >= 4:
                        code = code[0:4]
                    elif code[0] == 'E'and len(code) >= 3:
                        code = code[0:3]

                    if code in code_s2i:

                        claim_code_count += 1
                        claim_code.add(code_s2i[code])

                if len(claim_code) > 0:
                    patient_claim[cid] = claim_code
                    pre_pid = pid

            claims[pre_pid] = patient_claim
            print "Dumping data"
            pickle.dump(claims,open(output_pkl,"wb"))
            print "Total Patients: " + str(len(claims))
            print "Total Claims: " + str(lc)
            print "Total Code: " + str(claim_code_count)

# Total Patients: 85295
# Total Claims: 792562
# Total Code: 1973327
        except Exception as e:
            print "Fail to process "+ str(lc)
            print pid
            print e

    #Data Structure:
    #   pid -> ordered dict(
    #           cid1 -> set(numeric code)
    #           cid2 -> set(numeric code)
    #           ...
    #           )
    # r = pickle.load(open("claim.pkl","rb")) # print r
    # print


def preprocess_code(claim_csv,icd_json, output_pkl):
    lc = 0
    codes = set()
    with open(claim_csv) as infile:
        infile.next()
        try:
            for line in infile:
                lc = lc + 1
                info = line.split(",")
                pid = info[0]
                cid = info[1]
                if (lc % 30000 == 0):
                    print "Process Line "+ str(lc)
                    print "Pid: " +pid + " Cid: " + cid

                for idx in range(12,22):
                    code = info[idx].strip()
                    if code == "":
                        continue
                    if code.isdigit():
                        codes.add(code[0:3])
                    if code[0] == 'V':
                        codes.add(code[0:3])
                    if code[0] == 'E'and len(code) >= 4:
                        codes.add(code[0:4])
                    elif code[0] == 'E'and len(code) >= 3:
                        codes.add(code[0:3])


            code_list = []
            tree = ICD9(icd_json)
            for c in list(codes):
                if tree.find(c) is not None:
                    code_list.append(c)

            print "Distinct Code Count: ", len(code_list)

            code_i2s = {i:s for i,s in enumerate(code_list)}
            code_s2i = {s:i for i,s in enumerate(code_list)}

            code_info = dict()
            code_info["code_i2s"] = code_i2s
            code_info["code_s2i"] = code_s2i

            pickle.dump(code_info,open(output_pkl,"wb"))
        except Exception as e:
            print "Fail to process "+ str(lc)
            print pid
            print e


    # r = pickle.load(open("code.pkl","rb"))
    # print r


def preprocess_patient(patient_csv,output_pkl):

    patients = dict()
    with open(patient_csv) as infile:
        infile.next() #Exclude the header line
        lc = 0
        try:
            for line in infile:
                lc = lc + 1
                info = line.split(",")
                pid = info[0]
                if (lc % 10000 == 0):
                    print "Process Line "+ str(lc)
                    print "Pid: " +str(pid)
                try:
                    record = list()
                    birth = int(info[1])
                    death = info[2]
                    if death == "":
                        age = 2010 - birth / 10000
                    else:
                        age = (int(death) - int(birth)) / 10000

                    sex = 2 - int(info[3])
                    record.append(sex)
                    race = int(info[4]) / 5.0
                    record.append(race)
                    record.append(age / 100.0)
                    for choronic_ID in range(12,23):
                        record.append(2 - int(info[choronic_ID]))
                    patients[pid] = record
                except ValueError as e:
                   print "Ignore line " +str(lc) + " for pid: " + pid
            pickle.dump(patients, open(output_pkl,"wb"))
            print "Total Line: " + str(lc)
        except Exception as e:
            print "Fail to process "+ str(lc)
            print pid
            print e

if __name__ == "__main__":
    # preprocess_code("EMR/claim.csv", "icd9.json", "code.pkl")
    preprocess_claim("code.pkl", "EMR/claim.csv", "claim.pkl")
