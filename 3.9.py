from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata


def load_data_fashion_mnist(batch_size):
    transformer = gdata.vision.transforms.ToTensor()
    num_workers = 0
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    print(len(mnist_train), len(mnist_test))
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                  batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 1024, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(nd.dot(X, W1) + b1)
    H2 = relu(nd.dot(H1, W2) + b2)
    return nd.dot(H2, W3) + b3


loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 5, 0.5


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


