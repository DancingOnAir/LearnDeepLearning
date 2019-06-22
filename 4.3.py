from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10, ))
net.initialize(init=MyInit())

X = nd.random.uniform(shape=(2, 20))
Y = net(X)

X = X.reshape(shape=(1, 20))
Y = net(X)
#
# net.initialize(init=MyInit(), force_reinit=True)
