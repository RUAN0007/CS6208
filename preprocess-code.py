import cPickle as pickle

lc = 0
dcodes = set()
pcodes = set()
with open("data/claim.csv") as infile:
	infile.next()
	try:
		for line in infile:
			lc = lc + 1
			info = line.split(",")
			pid = info[0]
			cid = info[1]
			if (lc % 10000 == 0):
				print "Process Line "+ str(lc)
				print "Pid: " +pid + " Cid: " + cid

			for diagnosisId in range(12,22):
				dcode = info[diagnosisId].strip()[0:3]
				if dcode != "":
					dcodes.add(dcode)
			for procedureId in range(31,76):
				pcode = info[procedureId].strip()[0:3]
				if pcode != "":
					pcodes.add(pcode)

		print "Diacode Length: " + str(len(dcodes))

		dcode_list = list(dcodes)
		dcode_i2s = {i:s for i,s in enumerate(dcode_list)}
		dcode_s2i = {s:i for i,s in enumerate(dcode_list)}

		print "Procedure Code Length: " + str(len(pcodes))

		pcode_list = list(pcodes)
		pcode_i2s = {i:s for i,s in enumerate(pcode_list)} 
		pcode_s2i = {s:i for i,s in enumerate(pcode_list)} 
		print "Total Line: " + str(lc)

		code = dict()
		code["dcode_i2s"] = dcode_i2s
		code["dcode_s2i"] = dcode_s2i
		code["pcode_i2s"] = pcode_i2s
		code["pcode_s2i"] = pcode_s2i

		pickle.dump(code,open("code.pkl","wb"))
	except Exception as e:
		print "Fail to process "+ str(lc)
		print pid
		print e


# r = pickle.load(open("code.pkl","rb"))
# print r
import cPickle as pickle

lc = 0
dcodes = set()
pcodes = set()
with open("data/claim.csv") as infile:
	infile.next()
	try:
		for line in infile:
			lc = lc + 1
			info = line.split(",")
			pid = info[0]
			cid = info[1]
			if (lc % 10000 == 0):
				print "Process Line "+ str(lc)
				print "Pid: " +pid + " Cid: " + cid

			for diagnosisId in range(12,22):
				dcode = info[diagnosisId].strip()[0:3]
				if dcode != "":
					dcodes.add(dcode)
			for procedureId in range(31,76):
				pcode = info[procedureId].strip()[0:3]
				if pcode != "":
					pcodes.add(pcode)

		print "Diacode Length: " + str(len(dcodes))

		dcode_list = list(dcodes)
		dcode_i2s = {i:s for i,s in enumerate(dcode_list)}
		dcode_s2i = {s:i for i,s in enumerate(dcode_list)}

		print "Procedure Code Length: " + str(len(pcodes))

		pcode_list = list(pcodes)
		pcode_i2s = {i:s for i,s in enumerate(pcode_list)} 
		pcode_s2i = {s:i for i,s in enumerate(pcode_list)} 
		print "Total Line: " + str(lc)

		code = dict()
		code["dcode_i2s"] = dcode_i2s
		code["dcode_s2i"] = dcode_s2i
		code["pcode_i2s"] = pcode_i2s
		code["pcode_s2i"] = pcode_s2i

		pickle.dump(code,open("code.pkl","wb"))
	except Exception as e:
		print "Fail to process "+ str(lc)
		print pid
		print e


# r = pickle.load(open("code.pkl","rb"))
# print r
import cPickle as pickle

lc = 0
dcodes = set()
pcodes = set()
with open("data/claim.csv") as infile:
	infile.next()
	try:
		for line in infile:
			lc = lc + 1
			info = line.split(",")
			pid = info[0]
			cid = info[1]
			if (lc % 10000 == 0):
				print "Process Line "+ str(lc)
				print "Pid: " +pid + " Cid: " + cid

			for diagnosisId in range(12,22):
				dcode = info[diagnosisId].strip()[0:3]
				if dcode != "":
					dcodes.add(dcode)
			for procedureId in range(31,76):
				pcode = info[procedureId].strip()[0:3]
				if pcode != "":
					pcodes.add(pcode)

		print "Diacode Length: " + str(len(dcodes))

		dcode_list = list(dcodes)
		dcode_i2s = {i:s for i,s in enumerate(dcode_list)}
		dcode_s2i = {s:i for i,s in enumerate(dcode_list)}

		print "Procedure Code Length: " + str(len(pcodes))

		pcode_list = list(pcodes)
		pcode_i2s = {i:s for i,s in enumerate(pcode_list)} 
		pcode_s2i = {s:i for i,s in enumerate(pcode_list)} 
		print "Total Line: " + str(lc)

		code = dict()
		code["dcode_i2s"] = dcode_i2s
		code["dcode_s2i"] = dcode_s2i
		code["pcode_i2s"] = pcode_i2s
		code["pcode_s2i"] = pcode_s2i

		pickle.dump(code,open("code.pkl","wb"))
	except Exception as e:
		print "Fail to process "+ str(lc)
		print pid
		print e


