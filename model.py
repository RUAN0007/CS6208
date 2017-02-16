from singa import layer
from singa import metric
from singa import loss
from singa import device
from singa import net as ffnet
from singa import tensor
import numpy as np
import data

def create(distinct_code_count, code_embed_size, demo_feature_count,visit_embed_size, use_cpu=True):
    if use_cpu:
        layer.engine = 'singacpp'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy())
    wdense = layer.Dense('code_dense', code_embed_size, input_sample_shape=(distinct_code_count,))
    net.add(wdense)
    code_relu = layer.Activation('Code_RELU')
    net.add(code_relu)
    #dummy layer needs to explicitly setup
    demo_dummy = layer.Dummy("demo")
    demo_dummy.setup((demo_feature_count, ))
    net.add(demo_dummy, src = [])
    net.add(layer.Concat('visit_concat', 1, [(code_embed_size, ),(demo_feature_count, )]),src =[code_relu, demo_dummy])
    net.add(layer.Dense('visit_dense', visit_embed_size))
    net.add(layer.Activation('visit_RELU'))
    net.add(layer.Dense('output_dense', distinct_code_count))

#Init the weight from -1 to 1 and init bias to 1
    for (p, name) in zip(net.param_values(), net.param_names()):
        # print name, p.shape
        if "weight" in name:
            p.uniform(-1,1)
        elif "bias" in name:
            p.set_value(0)

    return net, wdense.param_values()[0]

