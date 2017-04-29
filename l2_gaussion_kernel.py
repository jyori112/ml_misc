import numpy as np
import matplotlib.pyplot as plt

def generate_data(N):
  x = np.linspace(-3, 3, N)
  pix = np.pi * x
  y = np.sin(pix)/pix+0.1*x+0.2*np.random.randn(N)
  return x, y

def main(h = 0.3, l=0.1):
  # Generate Data
  train_x, train_y = generate_data(100)

  # Precompute
  hh = 2*h**2

  # Compute matrix K
  K = np.exp(-(np.square(train_x-train_x[:,None])//hh))

  # Compute theta
  t = (np.linalg.inv(K.dot(K)+l*np.eye(K.shape[0])).dot(K)).dot(train_y)

  # Compute matrix K for test data
  test_x = np.linspace(-3, 3, 10000)
  test_K = np.exp(-(np.square(test_x-train_x[:,None])//hh))
  pred_y = test_K.T.dot(t)

  plt.plot(train_x, train_y, 'o')
  plt.plot(test_x, pred_y, '-')
  plt.show()

if __name__ == '__main__':
  main()
