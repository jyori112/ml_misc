import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy import spatial

def cosine_sim(x, y):
  return -x.dot(y.T)/(np.linalg.norm(x, axis=1)[:,None]*np.linalg.norm(y, axis=1))

def euclidean(x, y):
  return np.linalg.norm(x[:,None]-y, axis=2)

def knn(train_X, train_y, test_X, k, dist_func=euclidean):
  dist = dist_func(train_X, test_X)
  order = np.argsort(-dist, axis=0)[:k]
  pred_y = mode(train_y[order], axis=0)[0].flatten()
  return pred_y

def main():
  mnist = fetch_mldata('MNIST original')
  mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))
  mnist_X = mnist_X / 255.0
  train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

  train_X = train_X[:1000]
  test_X = test_X[:1000]
  train_y = train_y[:1000]
  test_y = test_y[:1000]

  pred_y = knn(train_X, train_y, test_X, k=3)
  print(f1_score(test_y_mini, pred_y, average='macro'))

if __name__ == '__main__':
  main()