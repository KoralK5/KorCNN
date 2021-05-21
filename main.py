print('\nImporting Modules')
import os
import numpy as np
from time import time
from matplotlib import pyplot as plt
from cv2 import imread, resize
from CNN import *
from train import *
from optimizers import *
np.random.seed(1)

def fix(img, dims):
	return np.array(resize(img, (dims[0],dims[1])).reshape(dims[0],dims[1]*dims[2])) / 255.0

def grab(path, dims):
	data = []
	categories = ['hem','all']
	folders = ['fold_0','fold_1','fold_2']
	for folder in folders:
		for category in categories:
			pt = f'{path}\\{folder}'
			p = f'{pt}\\{category}'
			output = int(category=='hem')
			for img in os.listdir(p):
				data.append([fix(imread(os.path.join(p,img)), dims), np.array([output, not(output)])])
	np.random.shuffle(data)
	return data

def plot(accuracy, loss, epoch_accuracy, epoch_loss, title='Model'):
	fig, axs = plt.subplots(2,2)
	fig.suptitle(title)
	
	axs[0,0].plot(range(len(epoch_accuracy)), epoch_accuracy, 'tab:blue')
	axs[0,0].set(ylabel='Accuracy')

	axs[1,0].plot(range(len(epoch_loss)), epoch_loss, 'tab:red')
	axs[1,0].set(xlabel='Epoch', ylabel='Loss')

	axs[0,1].plot(range(len(accuracy)), accuracy, 'tab:blue')
	axs[0,1].set()

	axs[1,1].plot(range(len(loss)), loss, 'tab:red')
	axs[1,1].set(xlabel='Iteration')

	plt.show()

def run(data, model, optimizer, path, rate=0.001, beta=0.9, scale=1, epochs=3):
	start = time()
	f = open(f'{path}model\\scores.txt', 'r+'); f.truncate(0)
	l, a, dims = [], [1,0], len(data)
	for epoch in range(epochs):
		for row in range(dims):
			output, loss, accuracy = train(data[row][0], data[row][1], model, optimizer, rate, beta, scale)
			l.append(loss); a.append(accuracy)

			np.save(f'{path}model\\weights.npy', np.array([model[2].weight, model[2].bias], dtype=object))
			f = open(f'{path}model\\scores.txt', 'a'); f.write(f'\n{loss}'); f.close()

			percent = int(((epoch)/epochs + ((row+1)/dims) * (1/epochs)) * 100)

			print(f'\n{percent*2//10*"▣"}▷ {percent}%')
			print(f'ITERATION {row+1}/{dims} OF EPOCH {epoch+1}/{epochs}:\n')
			print('Output    ➤ ', output)
			print('Real      ➤ ', data[row][1])
			print('Loss      ➤ ', '{0:.4f}'.format(loss))
			print('Accuracy  ➤ ', accuracy, f'({a[::-1].index(not(a[-1]))} streak)')
			print('Time      ➤ ', f'{int(time() - start)}s\n')

	print('\n\nTRAINING REPORT\n')
	print('Loss     ➤ ', sum(l)/len(l))
	print('Accuracy ➤ ', sum(a)/len(a))
	print('Duration ➤ ', '{0:.4f}'.format(time() - start), 'seconds')

	return a[2:], l

print('\nReading Data')

path = input('Model Path:')
dataPath = input('Data Path:')

dims = (96,96,3)
data = grab(dataPath, dims)

model = [
	Conv(18,7),
	Maxpool(4),
	Softmax(int((dims[0]/4-2)*(dims[1]*dims[2]/4-2)*18), data[0][1].size) # 22 * 70 * 18
	]

optimizer = gradient_descent
rate = 0.005
beta = 0.9
scale = 1
epochs = 3

print('\nTraining Model')

accuracy, loss = run(data, model, optimizer, path, rate=rate, beta=beta, scale=scale, epochs=epochs)

epoch_accuracy = np.sum(np.array_split(accuracy, epochs), axis=1) / len(data)
epoch_loss = np.sum(np.array_split(loss, epochs), axis=1) / len(data)

print('\n\nTRAINING PROGRESS\n')
print('Accuracy ➤ ', epoch_accuracy)
print('Loss     ➤ ', epoch_loss)

plot(accuracy, loss, epoch_accuracy, epoch_loss, 'TRAINING PROGRESS')
