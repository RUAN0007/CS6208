import cPickle as pk
import numpy as np
import sys


def prepare(claims, patients, max_claim_count,distinct_code_count, demo_feature_count, t_ratio=0.8, w=2):

	#Process data into matrix
	print "Allocating memory now: "
	#claim_data[i][j] = 1 if ith claim has medical code j.
	claim_data = np.zeros((max_claim_count, distinct_code_count),dtype=np.float32) #Shape: claim_count * claim_feature

	#patient_data[i][j]: the demographic features of patients of ith claim
	patient_data = np.zeros((max_claim_count,demo_feature_count),dtype=np.float32) #Shape:  claim_count * demo_feature

	ngb_claim_data = np.zeros((max_claim_count,distinct_code_count),dtype=np.float32)

	print "Start Processing Data: "
	claim_num = 0
	code_pair_num = 0
	terminate = False

	for pid, pt_claims in claims.iteritems():
		#pid: patient id
		#pt_claims: an ordered dict consisting of claim code

		for j,(cid, claim_codes) in enumerate(pt_claims.items()):
			code_pair_num += len(claim_codes) * (len(claim_codes) - 1)
			pt_claim_count = len(pt_claims)
			for n in range(j - w, j + w + 1):
				if n == j:
					continue
				if n < 0 or n >= pt_claim_count:
					continue

				if claim_num == max_claim_count:
					terminate = True
					break

				if claim_num % 10000 == 0:
					print "\tProcess Claim ", claim_num

				pt_records = patients[pid]
				for i,r in enumerate(pt_records):
					patient_data[claim_num,i] = r

				for c in claim_codes:
					claim_data[claim_num,c] = 1.0

				neighbor_codes = pt_claims.items()[n][1]

				for c in neighbor_codes:
					ngb_claim_data[claim_num,c] = 1.0
				claim_num += 1
			#End of for n neighbor
			if terminate:
				break
		#End of for j,(cid, claim_codes) in enumerate(pt_claims.items()):
		if terminate:
			break
	#End of for claims

	print "\tFinishing building claim data. Total Claim Num: ", claim_num

	code_data = np.zeros((code_pair_num, distinct_code_count),dtype=np.float32)
	code_labels = np.zeros(code_pair_num,dtype=np.int32)

	code_pair_idx = 0
	finish_code = False

	for pid, pt_claims in claims.iteritems():
		#pid: patient id
		#pt_claims: an ordered dict consisting of claim code
		for j,(cid, claim_codes) in enumerate(pt_claims.items()):
			if code_pair_idx == code_pair_num:
				finish_code = True
				break
			for code_idx1, code1 in enumerate(claim_codes):
				for code_idx2, code2 in enumerate(claim_codes):
					if code_idx1 == code_idx2:
						continue
					if code_pair_idx % 100000 == 0:
						print "Processing code pair ", code_pair_idx
					code_data[code_pair_idx, code1] = 1.0
					code_labels[code_pair_idx] = code2
					code_pair_idx += 1

		if finish_code:
			break

	print "\tFinishing building code data. Total Code Num: ",code_pair_idx

	claim_perm = np.random.permutation(claim_num)
	np.take(claim_data,claim_perm, axis=0,out=claim_data)
	np.take(patient_data,claim_perm, axis=0,out=patient_data)
	np.take(ngb_claim_data,claim_perm, axis=0,out=ngb_claim_data)

	code_perm = np.random.permutation(code_pair_num)
	np.take(code_data, code_perm, axis=0, out=code_data)
	np.take(code_labels, code_perm, axis=0, out=code_labels)
	print "Finish Random Shuffling: "


	train_claim_count = int(claim_num * t_ratio)
	test_claim_count = claim_num - train_claim_count

	train_claim_data = claim_data[0:train_claim_count,:]
	train_patient_data = patient_data[0:train_claim_count,:]
	train_ngb_claim_data = ngb_claim_data[0:train_claim_count,:]

	test_claim_data = claim_data[train_claim_count:,:]
	test_patient_data = patient_data[train_claim_count:,:]
	test_ngb_claim_data = ngb_claim_data[train_claim_count:,:]

	train_code_count = int(code_pair_num * t_ratio)
	test_code_count = claim_num - train_code_count

	train_code_data = code_data[0:train_code_count, :]
	train_code_labels = code_labels[0:train_code_count]

	test_code_data = code_data[train_code_count:,:]
	test_code_labels = code_labels[train_code_count:]

	print "Finish Partition training and testing data: "

	train_data = [train_claim_data, train_patient_data, train_ngb_claim_data, train_code_data, train_code_labels]

	test_data = [test_claim_data, test_patient_data, test_ngb_claim_data, test_code_data, test_code_labels]

	return train_data, test_data


if __name__ == "__main__":

	claims = pk.load(open("claim.pkl","rb"))
	# print "Len: ", len(claims)
	patients = pk.load(open("patient.pkl","rb"))

	prepare(claims, patients, max_claim_count=10000, distinct_code_count = 1722, demo_feature_count = 14, t_ratio=0.8, w=2)

