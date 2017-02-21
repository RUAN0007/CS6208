import sys, os, traceback
import glob, random, shutil, time
import numpy as np
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from singa import tensor, device, optimizer, layer
from singa import utils, initializer, metric, loss
from singa.proto import model_pb2
from rafiki.agent import Agent, MsgType
import cPickle as pk

import model
import data

# Global Parameters
# code_embed_size = 64
# visit_embed_size = 128
# distinct_code_count = 1722 #Num of medical codes
# demo_feature_count = 14 #Num of demo features
# patient_file = ""
# visit_file = ""
# max_claim_count = 10000

def main():

    '''Command line options'''
    argv = sys.argv
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train Alexnet over CIFAR10")

        parser.add_argument('-p', '--port', default=9999, help='listening port')
        parser.add_argument('-C', '--use_cpu', default=True)
        parser.add_argument('--max_epoch', default=140)

        parser.add_argument('--visit_file', default="claim.pkl", type=str, help='The path to the Pickled file containing visit information of patients')
        parser.add_argument('--n_input_codes',default=1722, type=int, help='The number of unique input medical codes')

        parser.add_argument('--patient_file',default='patient.pkl', type=str, help='The path to the Pickled file containing demographic features of patients')
        parser.add_argument('--n_demo_features', default=14, type=int, help='The number of demographic features')
        parser.add_argument('--code_size', type=int, default=64, help='The size of the code representation (default value: 64)')
        parser.add_argument('--visit_size', type=int, default=128, help='The size of the visit representation (default value: 128)')


        # Process arguments
        args = parser.parse_args()
        port = args.port
        use_cpu = args.use_cpu

        patient_file = args.patient_file
        visit_file = args.visit_file
        distinct_code_count = args.n_input_codes
        demo_feature_count = args.n_demo_features
        code_embed_size = args.code_size
        visit_embed_size = args.visit_size

        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            # dev = device.create_cuda_gpu()

        m,cdense = model.create(distinct_code_count,
                                code_embed_size,
                                demo_feature_count,
                                visit_embed_size,
                                use_cpu)

        # agent = Agent(port)
        train(m, cdense,
              visit_file, patient_file,
              distinct_code_count,
              demo_feature_count,
              code_embed_size,
              dev, args.max_epoch
              )
        # agent.stop()

    except SystemExit:
        return
    except:
        #p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


def handle_cmd(agent):
    pause = False
    stop = False
    while not stop:
        key, val = agent.pull()
        if key is not None:
            msg_type = MsgType.parse(key)
            if msg_type.is_command():
                if MsgType.kCommandPause.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = True
                elif MsgType.kCommandResume.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = False
                elif MsgType.kCommandStop.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    stop = True
                else:
                    agent.push(MsgType.kStatus, "Warning, unkown message type")
                    print "Unsupported command %s" % str(key)
        if pause and not stop:
            time.sleep(0.1)
        else:
            break
    return stop


def get_claim_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.001

def get_code_lr(epoch):
    return 0.05

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
    for b in range(num_train_claim_batch):
        train_claims_batch = train_claims[b * claim_batch_size:(b + 1) * claim_batch_size]
        train_patients_batch = train_patients[b * claim_batch_size:(b + 1) * claim_batch_size]
        train_claim_labels_batch = train_claim_labels[b * claim_batch_size:(b + 1) * claim_batch_size]

        t_claims.copy_from_numpy(train_claims_batch)
        t_patients.copy_from_numpy(train_patients_batch)
        t_labels.copy_from_numpy(train_claim_labels_batch)

        tx = {"code_dense": t_claims, "demo": t_patients}
        grads, (l, _) = claim_net.train(tx, t_labels)

        claim_loss += l
        for (s, p, g) in zip(claim_net.param_specs(), claim_net.param_values(), grads):
            opt.apply_with_lr(epoch, get_claim_lr(epoch), g, p, str(s.name))
        info = 'training claim_loss = %f' % (l)
        utils.update_progress(b * 1.0 / num_train_claim_batch, info)

    print '\n  training claim_loss = %f' % (claim_loss / num_train_claim_batch)

    claim_loss = 0.0
    for b in range(num_test_claim_batch):
        test_claims_batch = test_claims[b * claim_batch_size:(b + 1) * claim_batch_size]
        test_patients_batch = test_patients[b * claim_batch_size:(b + 1) * claim_batch_size]
        test_claim_labels_batch = test_claim_labels[b * claim_batch_size:(b + 1) * claim_batch_size]
        t_claims.copy_from_numpy(test_claims_batch)
        t_patients.copy_from_numpy(test_patients_batch)
        t_labels.copy_from_numpy(test_claim_labels_batch)

        tx = {"code_dense": t_claims, "demo": t_patients}
        l, _ = claim_net.evaluate(tx, t_labels)

        claim_loss += l
    print '    testing claim_loss = %f' % (claim_loss / num_test_claim_batch)




def train(claim_net, cdense_w,
          claim_path, patient_path,
          distinct_code_count,demo_feature_count,
          code_embed_size,
          dev,
          max_epoch=50,
          max_claim_count=10000,
          claim_batch_size=100,
          code_batch_size=500,
          agent=None):

    if agent is not None:
        agent.push(MsgType.kStatus, 'Start Loading data...')

    claims,patients = load_data(claim_path, patient_path)
    train_data, test_data = data.prepare(claims,
                                         patients,
                                         max_claim_count,
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

    if agent is not None:
        agent.push(MsgType.kStatus, 'Finish Loading data')

    t_claims = tensor.Tensor((claim_batch_size, distinct_code_count), dev)
    t_patients = tensor.Tensor((claim_batch_size, demo_feature_count), dev)
    t_labels = tensor.Tensor((claim_batch_size, distinct_code_count), dev)

    t_codes = tensor.Tensor((code_batch_size, distinct_code_count), dev)
    t_code_labels = tensor.Tensor((code_batch_size, ), dev)


    claim_net.to_device(dev)

    num_train_claim_batch = train_claims.shape[0] / claim_batch_size
    num_test_claim_batch = test_claims.shape[0] / claim_batch_size
    claim_batch_size_info = (claim_batch_size, num_train_claim_batch, num_test_claim_batch)

    num_train_code_batch = train_codes.shape[0] / code_batch_size
    num_test_code_batch = test_codes.shape[0] / code_batch_size

    opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)

    lossfun = loss.SoftmaxCrossEntropy()

    cdense1 = layer.Dense('code_dense1', code_embed_size, input_sample_shape=(distinct_code_count,))
    cdense1.to_device(dev)
    cdense2 = layer.Dense('code_dense2', distinct_code_count, input_sample_shape=(code_embed_size,))
    cdense2.to_device(dev)

    gcw = tensor.Tensor()
    gcw.reset_like(cdense1.param_values()[0])
    # print "cdense 1 w shape: ", cdense1.param_values()[0].shape #(1722, 64)
    # print "cdense 2 w shape: ", cdense2.param_values()[0].shape #(64,1722)

    for epoch in range(max_epoch):
        if agent is not None and handle_cmd(agent):
            break
        print 'Epoch %d' % epoch
        claim_tensors = (t_claims, t_patients, t_labels)
        train_claim_data = (train_claims, train_patients, train_claim_labels)
        test_claim_data = (test_claims, test_patients, test_claim_labels)
        train_claim(epoch, claim_net, opt, claim_tensors, train_claim_data, test_claim_data,claim_batch_size_info)

    #     if epoch > 0 and epoch % 10 == 0:
    #         claim_net.save('model_%d' % epoch)
    # claim_net.save('model')


        train_code_loss = 0.0
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

            batch_code_loss = lvalue.l1()
            train_code_loss += batch_code_loss

            grad = lossfun.backward()
            grad /= code_batch_size

            grad, (gw2,_) = cdense2.backward(model_pb2.kTrain, grad)
            _, (gw1,_) = cdense1.backward(model_pb2.kTrain, grad)

            cw1 = cdense1.param_values()[0]
            cw2 = cdense2.param_values()[0]

            cgw1 = tensor.eltwise_mult(gw1, tensor.sign(cw1))
            cgw2 = tensor.eltwise_mult(gw2, tensor.sign(cw2))

            # print "cw1 shape: ", cw1.shape, " gw1 shape: ", gw1.shape, " cgw1 shape: ", cgw1.shape #(1722, 64)
            # print "cw2 shape: ", cw2.shape, " gw2 shape: ", gw2.shape, " cgw2 shape: ", cgw2.shape # (64, 1722)
            # sys.exit()

            cgw2.to_host()
            ncgw2_t = np.transpose(tensor.to_numpy(cgw2))
            gcw.copy_from_numpy(ncgw2_t)
            cgw2.to_device(dev)

            gcw += cgw1
            gcw /= 2.0

            info = 'Batch training code loss = %f' % (batch_code_loss)
            # print "Batch: ", b
            # print "gw1: ", gw1.l1()
            # print "gw2: ", gw2.l1()
            # print "cgw1: ", cgw1.l1()
            # print "cgw2: ", cgw2.l1()
            # print "gcw: ", gcw.l1()
            # print "cdense_w: ", cdense_w.l1()
            # print info
            # print ""
            # input()


            opt.apply_with_lr(epoch, get_code_lr(epoch), gcw, cdense_w, "Fake Para")

            # sys.exit()
            utils.update_progress(b * 1.0 / num_train_code_batch, info)

        print '\n  training code_loss = %f' % (train_code_loss / num_train_code_batch)

        code_loss = 0.0
        for b in range(num_test_code_batch):

            t_codes.copy_from_numpy(test_codes[b * code_batch_size:(b + 1) * code_batch_size])
            t_code_labels.copy_from_numpy(test_code_labels[b * code_batch_size:(b + 1) * code_batch_size])

            cdense1out = cdense1.forward(model_pb2.kEval, t_codes)
            cdense2out = cdense2.forward(model_pb2.kEval, cdense1out)
            lvalue = lossfun.forward(model_pb2.kEval, cdense2out, t_code_labels)

            code_loss += lvalue.l1()

        print '    testing code_loss = %f' % (code_loss / num_test_code_batch)


        cdense_w.to_host()
        file_path = "embedding_code_%d" % (epoch)
        np.save(file_path, tensor.to_numpy(cdense_w))
        print "Save embedding code to %s. " % file_path
        cdense_w.to_device(dev)



if __name__ == '__main__':
    main()
