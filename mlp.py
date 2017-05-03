from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np

def softmax(x):
  y = np.exp(x)
  return y / np.sum(y, axis=1, keepdims=True)

def relu(x):
  y = x
  y[y<0] = 0
  return y

def derive_relu(x):
  y = np.zeros(shape=x.shape)
  y[x>0] = 1
  return y

def main():
  # Configuration
  n_epoch = 3  # How many epoch to run through?
  n_unit = 256  # Number of units in the hidden layer
  batch = 20    # Size of batch
  r = 0.1       # Learning Rate

  # Load Data
  mnist = fetch_mldata('MNIST original')
  mnist_x, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))
  mnist_x /= 255.0
  train_x, test_x, train_y, test_y = train_test_split(mnist_x, mnist_y, test_size=0.2)

  N = train_y.shape[0]

  # Convert y to one-host vector
  train_t = np.zeros(shape=(N, 10))
  train_t[np.arange(N), train_y] = 1
  test_t = np.zeros(shape=(test_y.shape[0], 10))
  test_t[np.arange(test_y.shape[0]), test_y] = 1

  # Initialize Weights
  init_sgm = 0.1
  W1 = init_sgm * np.random.randn(784, n_unit)
  b1 = init_sgm * np.random.randn(n_unit)
  W2 = init_sgm * np.random.randn(n_unit, 10)
  b2 = init_sgm * np.random.randn(10)

  print("N: ", N)
  # Iteration
  for epoch in range(n_epoch):
    loss_total = 0
    f1_total = 0

    for i in range(0, N, batch):
      # Batch
      batch_x = train_x[i:i+batch]
      batch_y = train_y[i:i+batch]
      batch_t = train_t[i:i+batch]

      # Feedforward
      u1 = batch_x.dot(W1) + b1
      h1 = relu(u1)
      u2 = h1.dot(W2) + b2
      y = softmax(u2)

      # Compute Loss
      loss_total -= np.sum(batch_t.dot(np.log(y.T)))
      f1_total += f1_score(np.argmax(u2, axis=1), batch_y, average='macro')

      # Compute Grad
      du2 = y-batch_t
      dW2 = h1[:, :, None] * du2[:, None, :]
      db2 = du2

      dh1 = du2.dot(W2.T)
      du1 = dh1 * derive_relu(u1)
      dW1 = batch_x[:, :, None] * du1[:, None, :]
      db1 = du1

      # Update

      W1 -= r * np.mean(dW1, axis=0)
      b1 -= r * np.mean(db1, axis=0)
      W2 -= r * np.mean(dW2, axis=0)
      b2 -= r * np.mean(db2, axis=0)

    # Validation
    u1 = test_x.dot(W1) + b1
    h1 = relu(u1)
    u2 = h1.dot(W2) + b2
    y = softmax(u2)

    # Compute Loss
    valid_loss = - np.mean(test_t.dot(np.log(y.T)))
    f1 = f1_score(np.argmax(u2, axis=1), test_y, average='macro')

    print("----- EPOCH: {} -----".format(epoch))
    print("LOST: {}".format(loss_total / N))
    print("F1:   {}".format(f1_total / (N//batch)))
    print("VAL LOST: {}".format(valid_loss))
    print("VAL F1:   {}".format(f1))

if __name__ == '__main__':
  main()