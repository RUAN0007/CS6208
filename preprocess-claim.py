import cPickle as pickle
import collections

code = pickle.load(open("code.pkl","rb"))

dcode_i2s = code["dcode_i2s"]
dcode_s2i = code["dcode_s2i"]
pcode_i2s = code["pcode_i2s"] 
pcode_s2i = code["pcode_s2i"] 

dcode_count = len(dcode_i2s)
pcode_count = len(pcode_i2s)

# patients = pickle.load(open("patient.pkl","rb"))

claims = {}
lc = 0
with open("data/claim.csv") as infile:
	infile.next()
	pre_pid = ""
	patient_claim = collections.OrderedDict()
	try:
		claim_dcode_count = 0
		claim_pcode_count = 0
		claim_code_count = 0
		for line in infile:
			lc = lc + 1
			if (lc % 10000 == 0):
				print "Process Line "+ str(lc)
				print "Pid: " +pid + " Cid: " + cid

			info = line.split(",")
			pid = info[0]

			# if not pid in patients:
			# 	print "At line " + str(lc)
			# 	print pid + " not in patient demo records"

			if pid != pre_pid and lc!=1:
				claims[pre_pid] = patient_claim
				patient_claim = collections.OrderedDict()

			cid = info[1]
			claim_code = set()


			for diagnosisId in range(12,22):
				dcode = info[diagnosisId].strip()[0:3]
				if dcode == "":
					continue
				claim_dcode_count += 1
				claim_code.add(dcode_s2i[dcode])

			for procedureId in range(31,76):
				pcode = info[procedureId].strip()[0:3]
				if pcode == "":
					continue 
				claim_pcode_count += 1
				claim_code.add(pcode_s2i[pcode] + dcode_count)


			claim_code_count = len(claim_code) + claim_code_count
			patient_claim[cid] = claim_code
			pre_pid = pid

		claims[pre_pid] = patient_claim	
		print "Dumping data"
		pickle.dump(claims,open("claim.pkl","wb"))	
		print "Total Patients: " + str(len(claims))
		print "Total Claims: " + str(lc)
		print "Total Diagnosis Code: " + str(claim_dcode_count)
		print "Total Procedure Code: " + str(claim_pcode_count)
		print "Total Code: " + str(claim_code_count)
	except Exception as e:
		print "Fail to process "+ str(lc)
		print pid
		print e

#Data Structure:
#	pid -> ordered dict(
#			cid1 -> set(numeric code)
#			cid2 -> set(numeric code)
#			... 
#			)
# r = pickle.load(open("claim.pkl","rb")) # print r
# print r