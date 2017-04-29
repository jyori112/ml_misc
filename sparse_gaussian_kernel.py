import numpy as np
import matplotlib.pyplot as plt

def generate_data(N):
  x = np.linspace(-3, 3, N)
  pix = np.pi * x
  y = np.sin(pix)/pix+0.1*x+0.2*np.random.randn(N)
  return x, y

def get_error(N, theta, hh):
  test_x = np.linspace(-3, 3, 1000)
  test_pix = np.pi*test_x
  test_y = np.sin(test_pix)/test_pix+0.1*test_x
  train_x = np.linspace(-3, 3, N)

  test_K = np.exp(-(np.square(test_x-train_x[:,None])//hh))
  pred_y = test_K.T.dot(theta)

  return np.linalg.norm(test_y-pred_y)

def main(h = 0.3, l=0.1):
  # Generate Data
  train_x, train_y = generate_data(100)

  # Precompute
  hh = 2*h**2

  # Compute matrix K
  K = np.exp(-(np.square(train_x-train_x[:,None])//hh))

  # Set first perameters by random
  theta = np.random.randn(K.shape[0])
  z = np.random.randn(K.shape[0])
  u = np.random.randn(K.shape[0])

  errors = np.zeros(shape=(100, ))

  for i in range(100):
    # Update
    theta = np.linalg.inv(K.dot(K)+np.eye(K.shape[0])).dot(K.dot(train_y)+z-u)
    z = np.maximum(0, theta+u-l)+np.minimum(0,theta+u+l)
    u = u + theta - z

    # Compute Error
    errors[i] = get_error(100, theta, hh)

  plt.hist(theta)
  plt.show()

  plt.plot(np.arange(100), errors, '-o')
  plt.show()

  test_x = np.linspace(-3, 3, 1000)
  test_pix = np.pi*test_x
  test_y = np.sin(test_pix)/test_pix+0.1*test_x

  test_K = np.exp(-(np.square(test_x-train_x[:,None])//hh))
  pred_y = test_K.T.dot(theta)
  plt.plot(train_x, train_y, 'o')
  plt.plot(test_x, pred_y, '-')
  plt.show()

if __name__ == '__main__':
  main()
