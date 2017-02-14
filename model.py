from singa import layer
from singa import metric
from singa import loss
from singa import device
from singa import net as ffnet
from singa import tensor
import numpy as np
import data

def create(total_code_count, code_embed_size, demo_feature_count,visit_embed_size, use_cpu=True):
    if use_cpu:
        layer.engine = 'singacpp'


    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy())

    net.add(layer.Dense('code_dense', code_embed_size, input_sample_shape=(total_code_count,) ) )
    code_relu = layer.Activation('Code_RELU')
    net.add(code_relu)
    #dummy layer needs to explicitly setup
    demo_dummy = layer.Dummy("demo")
    demo_dummy.setup((demo_feature_count, ))
    net.add(demo_dummy, src = [])
    net.add(layer.Concat('visit_concat', 1, [(code_embed_size, ),(demo_feature_count, )]),src =[code_relu, demo_dummy])
    net.add(layer.Dense('visit_dense', visit_embed_size))
    net.add(layer.Activation('visit_RELU'))
    net.add(layer.Dense('output_dense', total_code_count))

#Init the weight from -1 to 1 and init bias to 1
    for (p, name) in zip(net.param_values(), net.param_names()):
        print name, p.shape
        if "weight" in name:
            p.uniform(-1,1)
        elif "bias" in name:
            p.set_value(0)

    return net

if __name__ == '__main__':
    #Evaluate on some data on the created net
    # claims = pk.load(open("claim10000.pkl","rb"))
    # # print "Len: ", len(claims)
    # patients = pk.load(open("patient.pkl","rb"))
    # _, test_data = prepare(claims, patients, max_claim_count=10000, total_code_count = 1722, demo_feature_count = 14, t_ratio=0.99, w=2)

    # test_claims = test_data[0]
    # test_patients = test_data[0]
    # test_label = test_data[0]


    total_code_count = 1722
    code_embed_size = 128
    demo_feature_count = 14
    visit_embed_size = 128

    net = create(total_code_count=1722, code_embed_size = 128, demo_feature_count = 14,visit_embed_size = 128, use_cpu=True)

    cuda = device.get_default_device()

    net.to_device(cuda)

    claims_np = np.zeros((2, total_code_count), dtype =np.float32)
    claims_np[0,0] = 1.0
    claims_np[1, 100] = 1.0

    claims = tensor.from_numpy(claims_np)
    claims.to_device(cuda)

    patients = tensor.Tensor((2, demo_feature_count))
    patients.uniform(-1,1)
    patients.to_device(cuda)

    labels_np = np.zeros((2, total_code_count), dtype =np.float32)
    labels_np[0,0] = 1.0
    labels_np[1, 100] = 1.0

    labels = tensor.from_numpy(claims_np)
    labels.to_device(cuda)

    x = {"code_dense": claims, "demo": patients}

    print "Start Training"
    net.train(x, labels)
    print "End Training"
    # print "Start Forward"
    # net.forward(True,x)
    # print "End Forward"

    # print "Start Backward"
    # net.backward()
    # print "End Backward"
    # l.to_host()
    # l_np = tensor.to_numpy(l)

    # print "Loss: "
    # print l

    #create net and evaluate
    #print out the loss
