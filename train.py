import sys, os, traceback
import glob, random, shutil, time
import numpy as np
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from singa import tensor, device, optimizer
from singa import utils, initializer, metric
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
import cPickle as pk

import model
import data

# Global Parameters
code_embed_size = 64
visit_embed_size = 128
total_code_count = 1722 #Num of medical codes
demo_feature_count = 14 #Num of demo features
patient_file = ""
visit_file = ""
max_claim_count = 50000

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
	total_code_count = args.n_input_codes
	demo_feature_count = args.n_demo_features
	code_embed_size = args.code_size
	visit_embed_size = args.visit_size

        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            # dev = device.create_cuda_gpu()

        m = model.create(total_code_count, code_embed_size, demo_feature_count, visit_embed_size, use_cpu)

        agent = Agent(port)
        train(m, visit_file, patient_file, dev, agent, args.max_epoch)
        agent.stop()

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


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.001

def load_data(claim_path, patient_path):
    '''Load claims and patients from local pickle file'''

    claims = pk.load(open(claim_path,"rb"))
    patients = pk.load(open(patient_path,"rb"))
    return claims, patients

def train(net, claim_path, patient_path, dev, agent, max_epoch, batch_size=100):
    agent.push(MsgType.kStatus, 'Start Loading data...')

    claims,patients = load_data(claim_path, patient_path)
    train_data, test_data = data.prepare(claims, patients, max_claim_count,total_code_count, demo_feature_count)

    train_claims = train_data[0]
    train_patients = train_data[1]
    train_labels = train_data[2]

    test_claims = test_data[0]
    test_patients = test_data[1]
    test_labels = test_data[2]

    print "Train Claims shape: ", train_claims.shape
    print "Test Claims shape: ", test_claims.shape

    agent.push(MsgType.kStatus, 'Finish Loading data')

    t_claims = tensor.Tensor((batch_size, total_code_count), dev)
    t_patients = tensor.Tensor((batch_size, demo_feature_count), dev)
    t_labels = tensor.Tensor((batch_size, total_code_count), dev)

    net.to_device(dev)

    num_train_batch = train_claims.shape[0] / batch_size
    num_test_batch = test_claims.shape[0] / batch_size

    opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)

    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        print 'Epoch %d' % epoch

        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            test_claims_batch = test_claims[b * batch_size:(b + 1) * batch_size]
            test_patients_batch = test_patients[b * batch_size:(b + 1) * batch_size]
            test_labels_batch = test_labels[b * batch_size:(b + 1) * batch_size]
            t_claims.copy_from_numpy(test_claims_batch)
            t_patients.copy_from_numpy(test_patients_batch)
            t_labels.copy_from_numpy(test_labels_batch)

            tx = {"code_dense": t_claims, "demo": t_patients}
            l, _ = net.evaluate(tx, t_labels)

            loss += l
        print 'testing loss = %f' % (loss / num_test_batch)
        # put test status info into a shared queue
        info = dict(
            phase='test',
            step=epoch,
            accuracy=acc/num_test_batch,
            loss=loss/num_test_batch,
            timestamp=time.time())
        agent.push(MsgType.kInfoMetric, info)

        loss, acc = 0.0, 0.0
        for b in range(num_train_batch):
            train_claims_batch = train_claims[b * batch_size:(b + 1) * batch_size]
            train_patients_batch = train_patients[b * batch_size:(b + 1) * batch_size]
            train_labels_batch = train_labels[b * batch_size:(b + 1) * batch_size]

            t_claims.copy_from_numpy(train_claims_batch)
            t_patients.copy_from_numpy(train_patients_batch)
            t_labels.copy_from_numpy(train_labels_batch)

            tx = {"code_dense": t_claims, "demo": t_patients}
            grads, (l, _) = net.train(tx, t_labels)

            loss += l
            for (s, p, g) in zip(net.param_specs(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
            info = 'training loss = %f' % (l)
            utils.update_progress(b * 1.0 / num_train_batch, info)
        # put training status info into a shared queue
        info = dict(
            phase='train',
            step=epoch,
            accuracy=acc/num_train_batch,
            loss=loss/num_train_batch,
            timestamp=time.time())
        agent.push(MsgType.kInfoMetric, info)
        info = 'training loss = %f' % (loss / num_train_batch)
        print info
        if epoch > 0 and epoch % 10 == 0:
            net.save('model_%d' % epoch)
    net.save('model')


if __name__ == '__main__':
    main()
