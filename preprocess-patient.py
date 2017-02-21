'''
Created on Oct 8, 2016

@author: ruanpingcheng
'''
import cPickle as pickle

patients = dict()
with open("data/patient.csv") as infile:
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
        pickle.dump(patients, open("patient.pkl","wb"))
        print "Total Line: " + str(lc)
    except Exception as e:
        print "Fail to process "+ str(lc)
        print pid
        print e


