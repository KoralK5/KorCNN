import numpy as np
from CNN import *

def error_squared(p, q):
	return np.sum((p - q)**2)

def cross_entropy(p, q):
	return -sum([p[i]*np.log2(q[i]+1e-12) for i in range(len(p))])

def accuracy_eval(p, q):
	return np.argmax(p) == np.argmax(q)

def test(image, label, model):
	for layer in model:
		image = layer.forward_prop(image)

	loss = cross_entropy(image, label)
	accuracy = accuracy_eval(image, label)

	return image, loss, accuracy

def train(image, label, model, optimizer, rate=0.001, beta=0.9, scale=0.9):
	out, loss, acc = test(image, label, model)

	gradient = np.zeros(label.shape)
	gradient[np.argmax(label)] = -1 / out[np.argmax(label)]

	back = gradient
	for layer in model[::-1]:
		back = layer.back_prop(back, optimizer, rate, beta, scale)

	return out, loss, acc
