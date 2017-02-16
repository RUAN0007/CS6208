from singa import tensor, device, optimizer
import numpy as np

dev = device.get_default_device()
n = np.asarray([[1.0,-2.0], [-1.0,2.0], [4.0, 5.0]], dtype=np.float32)
t = tensor.from_numpy(n)
t_copy =

tt = t.T()
r = tensor.relu(t)
s = tensor.sign(r)
rs = tensor.eltwise_mult(r,s)

r.to_host()
s.to_host()
rs.to_host()
tt.to_host()

nr = tensor.to_numpy(r)
ns = tensor.to_numpy(s)
nrs = tensor.to_numpy(rs)
ntt = tensor.to_numpy(tt)

print "n"
print n

print "nr"
print nr

print "ns"
print ns

print "nrs"
print nrs

print "ntt"
print ntt
