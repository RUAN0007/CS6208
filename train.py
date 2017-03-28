import sys, os, traceback
import glob, random, shutil, time
import numpy as np
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from singa import tensor, device, optimizer, layer
from singa import utils, initializer, metric, loss
from singa import net as ffnet
from singa.proto import model_pb2
from singa.proto import core_pb2
import cPickle as pk

import model
import data
import visual

def main():

    '''Command line options'''
    argv = sys.argv
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Med2vec in Singa")

        parser.add_argument('--max_epoch', default=50)

        parser.add_argument('--visit_file', default="data/claim.pkl", type=str, help='The path to the Pickled file containing visit information of patients')
        parser.add_argument('--n_input_codes',default=1050, type=int, help='The number of unique input medical codes')

        parser.add_argument('--patient_file',default='data/patient.pkl', type=str, help='The path to the Pickled file containing demographic features of patients')
        parser.add_argument('--n_demo_features', default=14, type=int, help='The number of demographic features')
        parser.add_argument('--code_size', type=int, default=64, help='The size of the code representation (default value: 64)')
        parser.add_argument('--visit_size', type=int, default=128, help='The size of the visit representation (default value: 128)')

        # Process arguments
        args = parser.parse_args()

        patient_file = args.patient_file
        visit_file = args.visit_file
        distinct_code_count = args.n_input_codes
        demo_feature_count = args.n_demo_features
        code_embed_size = args.code_size
        visit_embed_size = args.visit_size

        use_cpu = True
        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        # else:
        #     print "runing with gpu"
            # dev = device.create_cuda_gpu()

        m,cdense = model.create(distinct_code_count,
                                code_embed_size,
                                demo_feature_count,
                                visit_embed_size,
                                use_cpu)

        claim_raw,patient_raw = load_data(visit_file, patient_file)

        (sub_claim_train_losses, sub_claim_test_losses, sub_claim_train_recall, sub_claim_test_recall, claim_train_losses, claim_test_losses, claim_train_recalls, claim_test_recalls, code_train_losses, code_test_losses, code_train_recalls, code_test_recalls) = train(m, cdense,
                  claim_raw, patient_raw,
                  distinct_code_count,
                  demo_feature_count,
                  code_embed_size,
                  dev,
                  11, #args.max_epoch,
                  )

        stat = dict()
        stat["sub_claim_train_loss"] = sub_claim_train_losses
        stat["sub_claim_test_loss"] = sub_claim_test_losses

        stat["sub_claim_train_recall"] = sub_claim_train_recall
        stat["sub_claim_test_recall"] = sub_claim_test_recall

        stat["claim_train_loss"] = claim_train_losses
        stat["claim_test_loss"] = claim_test_losses

        stat["claim_train_recall"] = claim_train_recalls
        stat["claim_test_recall"] = claim_test_recalls

        stat["code_train_loss"] = code_train_losses
        stat["code_test_loss"] = code_test_losses

        # stat["code_train_recall"] = code_train_recalls
        # stat["code_test_recall"] = code_test_recalls

        pk.dump(stat, open("stat.pkl","wb"))

    except SystemExit:
        return
    except:
        #p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2

def get_claim_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.0001

def get_code_lr(epoch):
    return 0.001

def load_data(claim_path, patient_path):
    '''Load claims and patients from local pickle file'''

    claims = pk.load(open(claim_path,"rb"))
    patients = pk.load(open(patient_path,"rb"))
    return claims, patients


def train_claim(epoch, claim_net, opt,
                claim_tensors, train_claim_data,
                test_claim_data, batch_size_info):
    (t_claims, t_patients, t_labels) = claim_tensors
    (train_claims, train_patients, train_claim_labels) = train_claim_data
    (test_claims, test_patients, test_claim_labels) = test_claim_data
    (claim_batch_size, num_train_claim_batch, num_test_claim_batch) = batch_size_info

    claim_loss = 0.0
    recall = 0.0
    for b in range(num_train_claim_batch):
        train_claims_batch = train_claims[b * claim_batch_size:(b + 1) * claim_batch_size]
        train_patients_batch = train_patients[b * claim_batch_size:(b + 1) * claim_batch_size]
        train_claim_labels_batch = train_claim_labels[b * claim_batch_size:(b + 1) * claim_batch_size]

        t_claims.copy_from_numpy(train_claims_batch)
        t_patients.copy_from_numpy(train_patients_batch)
        t_labels.copy_from_numpy(train_claim_labels_batch)

        tx = {"code_dense": t_claims, "demo": t_patients}
        grads, (l, r) = claim_net.train(tx, t_labels)

        claim_loss += l
        recall += r
        for (s, p, g) in zip(claim_net.param_specs(), claim_net.param_values(), grads):
            opt.apply_with_lr(epoch, get_claim_lr(epoch), g, p, str(s.name))
        # info = 'training claim_loss = %f' % (l)
        # utils.update_progress(b * 1.0 / num_train_claim_batch, info)

    train_loss = claim_loss / num_train_claim_batch
    train_recall = recall / num_train_claim_batch
    # print '\n  training claim_loss = %f, recall = %f' % (claim_loss / num_train_claim_batch, recall / num_train_claim_batch)

    claim_loss = 0.0
    recall = 0.0
    for b in range(num_test_claim_batch):
        test_claims_batch = test_claims[b * claim_batch_size:(b + 1) * claim_batch_size]
        test_patients_batch = test_patients[b * claim_batch_size:(b + 1) * claim_batch_size]
        test_claim_labels_batch = test_claim_labels[b * claim_batch_size:(b + 1) * claim_batch_size]
        t_claims.copy_from_numpy(test_claims_batch)
        t_patients.copy_from_numpy(test_patients_batch)
        t_labels.copy_from_numpy(test_claim_labels_batch)

        tx = {"code_dense": t_claims, "demo": t_patients}
        l, r = claim_net.evaluate(tx, t_labels)
        recall += r
        claim_loss += l
    # print '    testing claim_loss = %f, recall = %f' % (claim_loss / num_test_claim_batch, recall / num_test_claim_batch)

    test_loss = claim_loss / num_test_claim_batch
    test_recall = recall / num_test_claim_batch

    return (train_loss, test_loss), (train_recall, test_recall)


def train(claim_net, cdense_w,
          claim_raw, patient_raw,
          distinct_code_count,demo_feature_count,
          code_embed_size,
          dev,
          max_epoch=50,
          max_patient=1700,
          claim_batch_size=100,
          code_batch_size=500 ):

    print "Allocating Memory"
    data_buffer = []
    data_buffer.append(np.zeros((data.max_claim, distinct_code_count),dtype=np.float32))
    data_buffer.append(np.zeros((data.max_claim, demo_feature_count),dtype=np.float32))
    data_buffer.append(np.zeros((data.max_claim, distinct_code_count),dtype=np.int32))

    data_buffer.append(np.zeros((data.max_code_pair, distinct_code_count),dtype=np.float32))
    data_buffer.append(np.zeros(data.max_code_pair,dtype=np.int32))


    t_claims = tensor.Tensor((claim_batch_size, distinct_code_count), dev)
    t_patients = tensor.Tensor((claim_batch_size, demo_feature_count), dev)
    t_labels = tensor.Tensor((claim_batch_size, distinct_code_count), dev, core_pb2.kInt)

    t_codes = tensor.Tensor((code_batch_size, distinct_code_count), dev)
    t_code_labels = tensor.Tensor((code_batch_size, ), dev, core_pb2.kInt)

    claim_net.to_device(dev)

    opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)

    lossfun = loss.SoftmaxCrossEntropy()
    recallfun = metric.Recall(top_k=100)

    cdense1 = layer.Dense('code_dense1', code_embed_size, input_sample_shape=(distinct_code_count,))
    cdense1.to_device(dev)
    cdense2 = layer.Dense('code_dense2', distinct_code_count, input_sample_shape=(code_embed_size,))
    cdense2.to_device(dev)

    gcw = tensor.Tensor()
    gcw.reset_like(cdense1.param_values()[0])

    claim_train_losses = []
    claim_test_losses = []

    code_train_losses = []
    code_test_losses = []

    claim_train_recalls = []
    claim_test_recalls = []

    code_train_recalls = []
    code_test_recalls = []

    sub_claim_train_losses = []
    sub_claim_test_losses = []

    sub_claim_train_recalls = []
    sub_claim_test_recalls = []

    num_subepoch = len(claim_raw) / max_patient
    # num_subepoch = 2

    print "Number of Subepoch: ", num_subepoch

    for epoch in range(max_epoch):

        print "\nEpoch %d: " % (epoch + 1)
        if epoch % 5 == 0:
            cdense_w.to_host()
            visual.output_json(epoch,
                               tensor.to_numpy(tensor.relu(cdense_w)),
                               "embed/epoch%d.json" % epoch)
            cdense_w.to_device(dev)

        epoch_train_code_loss = 0.0
        epoch_train_code_recall = 0.0

        epoch_test_code_loss = 0.0
        epoch_test_code_recall = 0.0

        epoch_train_claim_loss = 0.0
        epoch_train_claim_recall = 0.0

        epoch_test_claim_loss = 0.0
        epoch_test_claim_recall = 0.0

        for subepoch in range(num_subepoch):
            # print "\n  Subepoch %d: " % (subepoch + 1)
            sub_train_code_loss = 0.0
            sub_train_code_recall = 0.0

            sub_test_code_loss = 0.0
            sub_test_code_recall = 0.0

            sub_train_claim_loss = 0.0
            sub_train_claim_recall = 0.0

            sub_test_claim_loss = 0.0
            sub_test_claim_recall = 0.0

            # prepare train_data and test_data for subepoch
            train_data, test_data = data.prepare(subepoch, max_patient,
                                                 claim_raw, patient_raw,
                                                 data_buffer,
                                                 distinct_code_count,
                                                 demo_feature_count)


            train_claims = train_data[0]
            train_patients = train_data[1]
            train_claim_labels = train_data[2]
            train_codes = train_data[3]
            train_code_labels = train_data[4]

            test_claims = test_data[0]
            test_patients = test_data[1]
            test_claim_labels = test_data[2]
            test_codes = test_data[3]
            test_code_labels = test_data[4]

            num_train_claim_batch = train_claims.shape[0] / claim_batch_size
            num_test_claim_batch = test_claims.shape[0] / claim_batch_size
            claim_batch_size_info = (claim_batch_size, num_train_claim_batch, num_test_claim_batch)

            num_train_code_batch = train_codes.shape[0] / code_batch_size
            num_test_code_batch = test_codes.shape[0] / code_batch_size

            claim_tensors = (t_claims, t_patients, t_labels)
            train_claim_data = (train_claims, train_patients, train_claim_labels)
            test_claim_data = (test_claims, test_patients, test_claim_labels)

            (sub_train_claim_loss, sub_test_claim_loss), (sub_train_claim_recall, sub_test_claim_recall) = train_claim(epoch, claim_net, opt, claim_tensors, train_claim_data, test_claim_data,claim_batch_size_info)

            for b in range(num_train_code_batch):

                #set W of cdense1 and cdense2 identical to cdense_w
                #zero b of cdense1 and cdense2

                cdense1.param_values()[0].copy_data(tensor.relu(cdense_w))
                cdense1.param_values()[1].set_value(0)

                cdense_w.to_host()
                ncw = tensor.to_numpy(tensor.relu(cdense_w))
                cdense_w.to_device(dev)

                cdense2.param_values()[0].copy_from_numpy(np.transpose(ncw))
                cdense2.param_values()[1].set_value(0)

                t_codes.copy_from_numpy(train_codes[b * code_batch_size:(b + 1) * code_batch_size])
                t_code_labels.copy_from_numpy(train_code_labels[b * code_batch_size:(b + 1) * code_batch_size])

                cdense1out = cdense1.forward(model_pb2.kTrain, t_codes)
                cdense2out = cdense2.forward(model_pb2.kTrain, cdense1out)

                lvalue = lossfun.forward(model_pb2.kTrain, cdense2out, t_code_labels)
                # recall = recallfun.evaluate(cdense2out, t_code_labels)

                batch_code_loss = lvalue.l1()
                sub_train_code_loss += batch_code_loss
                # sub_train_code_recall += recall

                grad = lossfun.backward()
                grad /= code_batch_size

                grad, (gw2,_) = cdense2.backward(model_pb2.kTrain, grad)
                _, (gw1,_) = cdense1.backward(model_pb2.kTrain, grad)

                cw1 = cdense1.param_values()[0]
                cw2 = cdense2.param_values()[0]

                cgw1 = tensor.eltwise_mult(gw1, tensor.sign(cw1))
                cgw2 = tensor.eltwise_mult(gw2, tensor.sign(cw2))

                cgw2.to_host()
                ncgw2_t = np.transpose(tensor.to_numpy(cgw2))
                gcw.copy_from_numpy(ncgw2_t)
                cgw2.to_device(dev)

                gcw += cgw1
                gcw /= 2.0

                opt.apply_with_lr(epoch, get_code_lr(epoch), gcw, cdense_w, "Fake Para")

                # sys.exit()

            for b in range(num_test_code_batch):

                t_codes.copy_from_numpy(test_codes[b * code_batch_size:(b + 1) * code_batch_size])
                t_code_labels.copy_from_numpy(test_code_labels[b * code_batch_size:(b + 1) * code_batch_size])

                cdense1out = cdense1.forward(model_pb2.kEval, t_codes)
                cdense2out = cdense2.forward(model_pb2.kEval, cdense1out)
                lvalue = lossfun.forward(model_pb2.kEval, cdense2out, t_code_labels)
                # recall = recallfun.evaluate(cdense2out, t_code_labels)

                sub_test_code_loss += lvalue.l1()
                # sub_test_code_recall += recall

            sub_train_code_loss = sub_train_code_loss / num_train_code_batch
            sub_train_code_recall = sub_train_code_recall / num_train_code_batch

            sub_test_code_loss = sub_test_code_loss / num_test_code_batch
            sub_test_code_recall = sub_test_code_recall / num_test_code_batch

            ###########Increment the epoch counter

            sub_claim_train_losses.append(sub_train_claim_loss)
            sub_claim_test_losses.append(sub_test_claim_loss)

            sub_claim_train_recalls.append(sub_train_claim_recall)
            sub_claim_test_recalls.append(sub_test_claim_recall)


            epoch_train_claim_loss += sub_train_claim_loss
            epoch_train_claim_recall += sub_train_claim_recall

            epoch_test_claim_loss += sub_test_claim_loss
            epoch_test_claim_recall += sub_test_claim_recall

            epoch_train_code_loss += sub_train_code_loss
            epoch_train_code_recall += sub_train_code_recall

            epoch_test_code_loss += sub_test_code_loss
            epoch_test_code_recall += sub_test_code_recall

            info = "\nVisit Loss: %5.3f (%5.3f), Recall: %5.3f (%5.3f)" % (sub_train_claim_loss, sub_test_claim_loss, sub_train_claim_recall, sub_test_claim_recall)

            info += "\tCode Loss: %5.3f (%5.3f), Recall: %5.3f (%5.3f)" % (sub_train_code_loss, sub_test_code_loss, sub_train_code_recall, sub_test_code_recall)

            utils.update_progress((subepoch + 1) * 1.0 / num_subepoch, info)

        # end of subepoch

        claim_train_losses.append(epoch_train_claim_loss / num_subepoch)
        claim_test_losses.append(epoch_test_claim_loss / num_subepoch)

        claim_train_recalls.append(epoch_train_claim_recall / num_subepoch)
        claim_test_recalls.append(epoch_test_claim_recall / num_subepoch)

        code_train_losses.append(epoch_train_code_loss / num_subepoch)
        code_test_losses.append(epoch_test_code_loss / num_subepoch)

        code_train_recalls.append(epoch_train_code_recall / num_subepoch)
        code_test_recalls.append(epoch_test_code_recall / num_subepoch)

        info = "\nVisit Loss: %5.3f (%5.3f), Recall: %5.3f (%5.3f)" % (claim_train_losses[-1], claim_test_losses[-1], claim_train_recalls[-1], claim_test_recalls[-1])

        info += "\nCode Loss: %5.3f (%5.3f), Recall: %5.3f (%5.3f)" % (code_train_losses[-1], code_test_losses[-1], code_train_recalls[-1], code_test_recalls[-1])
        print "Epoch %d:\n %s\n" % (epoch + 1, info)
    # end of epoch

    return sub_claim_train_losses, sub_claim_test_losses, sub_claim_train_recalls, sub_claim_test_recalls, claim_train_losses, claim_test_losses, claim_train_recalls, claim_test_recalls, code_train_losses, code_test_losses, code_train_recalls, code_test_recalls


if __name__ == '__main__':
    # ffnet.verbose = True
    main()
