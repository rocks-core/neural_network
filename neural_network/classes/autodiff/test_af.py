import tensorflow as tf
from TreeNodes.OperationNode import *
from neural_network.classes.autodiff import AutodiffFramework

if __name__ == '__main__':
	npw = np.random.random(size=(4, 4))
	npx = np.random.random(size=(4, 4))
	npz = np.random.random(size=(4, 8))

	af = AutodiffFramework(strict=True)

	v1 = af.add_variable(npx)

	b = af.stack([v1, npw], ax=0)
	a = b.compute()
	c = af.matmul(b, af.stack([af.transpose(v1), v1], ax=0))
	# d = af.matmul(c, npz)
	# e = af.product(d, npz)
	# f = af.get(e, (1, slice(None, None, None)))
	# a = e.compute()
	# h = af.pow(e, 2)
	# f = af.concat(e, e, ax=0)
	# a = f.compute()

	g1 = af.gradient(c, v1)

	x = tf.Variable(npx)
	w = tf.constant(npw)
	z = tf.constant(npz)
	with tf.GradientTape() as tape:
		y = tf.stack([x, w], axis=0)
		y = tf.matmul(y, tf.stack([tf.transpose(x), x], axis=0))
		# y = tf.linalg.matmul(w + x, tf.transpose(x))
		# y = tf.linalg.matmul(y, z)
		# y = tf.multiply(y, z)
		# y = tf.pow(y, 2)
		# y = tf.concat([y, y], axis=0)
		# print(y - a)

	g2 = tape.gradient(y, x)
	print(g2 - g1)


