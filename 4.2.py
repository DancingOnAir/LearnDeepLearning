from mxnet import nd, init
from mxnet.gluon import nn, loss as gloss
from mxnet import autograd


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
Y = net(X)

# print(X)
# print(net[0].params, type(net[0].params))
# print(net[1].params, type(net[1].params))
# print(net[0].params['dense0_weight'] == net[0].weight)
# print(net[0].weight.data())
# print(net[0].weight.grad())
# print(net[1].bias.data())
# print(net.collect_params())
# print(net.collect_params('.*weight'))


# X.attach_grad()
# with autograd.record():
#     y_hat = net(X)
# y_hat.backward()
# print(net[0].weight.grad())

net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])
net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])


net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
print(net(X))
print(net.collect_params())
print(net[1].weight.data() == net[2].weight.data())
