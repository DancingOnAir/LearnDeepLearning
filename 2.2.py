from mxnet import nd

x = nd.arange(12)
print(x)

print(x.shape)
print(x.size)

X = x.reshape((3, 4))
print(X)

print(nd.zeros((2, 3, 4)))
print(nd.ones((2, 3, 4)))

Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

print(nd.random.normal(0, 1, shape=(3, 4)))

print(X + Y)
print(X / Y)
print(X * Y)
print(X.exp())
print(nd.dot(X, Y.T))
print(nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1))
print(X == Y)
print(X.sum())
print(X.sum().asscalar())

A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
print(A + B)

previous = id(Y)
# Y = Y + X
Y += X
print(id(Y) == previous)

Z = nd.arange(4).reshape(2, 2)
print(Z)
print(Z.norm())

