import argparse
import io
import sys
import cPickle as pk
import numpy as np



def prepare(claims, patients, max_claim_count,total_code_count, demo_feature_count, t_ratio=0.8, w=2): 
		
	#Process data into matrix
	print "Allocating memory now: "	
	#claim_data[i][j] = 1 if ith claim has medical code j.
	claim_data = np.zeros((max_claim_count, total_code_count),dtype=np.float32) #Shape: claim_count * claim_feature

	#patient_data[i][j]: the demographic features of patients of ith claim 
	patient_data = np.zeros((max_claim_count,demo_feature_count),dtype=np.float32) #Shape:  claim_count * demo_feature

	neighbor_data = np.zeros((max_claim_count,total_code_count),dtype=np.float32)

	print "Start Processing Data: "	
	claim_num = 0
	terminate = False
	for pid, pt_claims in claims.iteritems():
		#pid: patient id
		#pt_claims: an ordered dict consisting of claim code
	
		for j,(cid, claim_codes) in enumerate(pt_claims.items()):

			pt_claim_count = len(pt_claims)
			for n in range(j - w, j + w + 1):
				if n == j:
					continue
				if n < 0 or n >= pt_claim_count:
					continue

				if claim_num == max_claim_count:
					terminate = True
					break

				if claim_num % 5000 == 0:
					print "\tProcess Claim ", claim_num

				pt_records = patients[pid]
				for i,r in enumerate(pt_records):
					patient_data[claim_num,i] = r	

				for c in claim_codes:	
					claim_data[claim_num,c] = 1.0

				neighbor_codes = pt_claims.items()[n][1]

				for c in neighbor_codes:
					neighbor_data[claim_num,c] = 1.0
				claim_num += 1
			#End of for n neighbor
			if terminate:
				break
		#End of for j,(cid, claim_codes) in enumerate(pt_claims.items()):
		if terminate:
			break
	#End of for claims
	
	print "\tFinishing building data. Total Claim Num: ", claim_num		
	print "Start Random Shuffling: "
	perm = np.random.permutation(claim_num)
	np.take(claim_data,perm, axis=0,out=claim_data)
	np.take(patient_data,perm, axis=0,out=patient_data)
	np.take(neighbor_data,perm, axis=0,out=neighbor_data)
		
	print "Start Partition training and testing data: "		
	train_sample_count = int(claim_num * t_ratio)
	test_sample_count = claim_num - train_sample_count
	
	train_claim_data = claim_data[0:train_sample_count,:]
	train_patient_data = patient_data[0:train_sample_count,:]
	train_neighbor_data = neighbor_data[0:train_sample_count,:]

	test_claim_data = claim_data[train_sample_count:,:]
	test_patient_data = patient_data[train_sample_count:,:]
	test_neighbor_data = neighbor_data[train_sample_count:,:]

	return [train_claim_data, train_patient_data, train_neighbor_data],[test_claim_data, test_patient_data, test_neighbor_data]

	#End for epoch

				
if __name__ == "__main__":

	claims = pk.load(open("claim10000.pkl","rb"))	
	# print "Len: ", len(claims)
	patients = pk.load(open("patient.pkl","rb"))
	# print "Len: ", len(patients)
	# small_claims = dict()
	# size = 10000
	# i = 0
	# for pid, pt_claims in claims.iteritems():
	# 	if i % 1000 == 0:
	# 		print i
	# 	if i == size:
	# 		break
	# 	small_claims[pid] = pt_claims
	# 	i += 1

	# pk.dump(small_claims, open("claim10000.pkl","wb"))
	prepare(claims, patients, max_claim_count=40000, total_code_count = 1722, demo_feature_count = 14, t_ratio=0.8, w=2)

		
