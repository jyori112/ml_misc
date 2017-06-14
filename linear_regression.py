import numpy as np

def generate_data():
    S = np.random.rand(10, 10)
    u = np.random.normal(size=(1000, 10))
    data = np.matmul(u, S)
    x = data[:, :9]
    y = data[:, 9]

    return x[:900], y[:900], x[900:], y[900:]


def main():
1   # Genarate Data
    train_x, train_y, test_x, test_y = generate_data()

    # Compute W
    w = np.linalg.pinv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)

    # Make prediction
    pred_y = test_x.dot(w)

    # Evaluate result
    print(np.mean(np.square(pred_y-test_y)))

if __name__ == '__main__':
    main()


