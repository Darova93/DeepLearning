import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets, load_cat_dataset


def layer_sizes(x, y):
    n_x = x.shape[0]
    n_h = 4
    n_y = y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    return {"W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2}


def forward_propagation(x, params):
    w1 = params["W1"]
    b1 = params["b1"]
    w2 = params["W2"]
    b2 = params["b2"]

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    assert (a2.shape == (1, x.shape[1]))

    cache = {"Z1": z1,
             "A1": a1,
             "Z2": z2,
             "A2": a2}

    return a2, cache


def compute_cost(a2, y):
    m = y.shape[1]

    log_probabilities = y * np.log(a2) + (1 - y) * np.log(1 - a2)
    cost = - np.sum(log_probabilities) / m

    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))

    return cost


def backward_propagation(params, cache, x, y):
    m = x.shape[1]

    w1 = params["W1"]
    w2 = params["W2"]

    a1 = cache["A1"]
    a2 = cache["A2"]

    d_z2 = a2 - y
    d_w2 = (1 / m) * np.dot(d_z2, a1.T)
    d_b2 = (1 / m) * np.sum(d_z2, axis=1, keepdims=True)
    d_z1 = w2.T * d_z2 * (1 - np.power(a1, 2))
    d_w1 = (1 / m) * np.dot(d_z1, x.T)
    d_b1 = (1 / m) * np.sum(d_z1, axis=1, keepdims=True)

    return {"dW1": d_w1,
            "db1": d_b1,
            "dW2": d_w2,
            "db2": d_b2}


def update_parameters(params, grads, learning_rate=0.01):
    w1 = params["W1"]
    b1 = params["b1"]
    w2 = params["W2"]
    b2 = params["b2"]

    d_w1 = grads["dW1"]
    d_b1 = grads["db1"]
    d_w2 = grads["dW2"]
    d_b2 = grads["db2"]

    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * d_b1
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * d_b2

    return {"W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2}


def nn_model(x, y, n_h, num_iterations=10000, learning_rate=0.01, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(x, y)[0]
    n_y = layer_sizes(x, y)[2]
    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        a2, cache = forward_propagation(x, params)
        cost = compute_cost(a2, y)
        grads = backward_propagation(params, cache, x, y)
        params = update_parameters(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return params


def predict(params, x):
    a2, cache = forward_propagation(x, params)
    return (a2 > 0.5) * 1.


def print_decision_boundary(input_data, label_data, params):
    plot_decision_boundary(lambda e: predict(params, e.T), input_data, label_data)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()


def sklearn_logistic_regression(x, y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(x.T, y.T)

    plot_decision_boundary(lambda e: clf.predict(e), x, y)
    plt.title("Logistic Regression")
    plt.show()

    lr_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, lr_predictions) + np.dot(1 - Y, 1 - lr_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")


def test_hidden_layer_size(hidden_layer_sizes):
    plt.figure(figsize=(16, 32))
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


def evaluate_accuracy(data, labels, params):
    predictions = predict(params, data)
    print('Accuracy: %d ' % float(
        (np.dot(labels, predictions.T) + np.dot(1-labels, 1 - predictions.T)) / float(labels.size) * 100)
          + '%'
    )


def load_data_sets(dataset, visualize_data):
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    planar_dataset = load_planar_dataset()
    cat_dataset = load_cat_dataset()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles,
                "planar_dataset": planar_dataset,
                "cat_dataset": cat_dataset}

    x, y = datasets[dataset]
    if dataset != "planar_dataset" and dataset != "cat_dataset":
        x, y = x.T, y.reshape(1, y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        y = y % 2

    # Visualize the data
    if visualize_data:
        plt.scatter(x[0, :], x[1, :], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    return x, y


if __name__ == '__main__':
    np.random.seed(1)  # set a seed so that the results are consistent

    # sklearn_logistic_regression(X, Y)
    X, Y = load_data_sets("planar_dataset", True)
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.12, print_cost=True)
    print_decision_boundary(X, Y, parameters)
    evaluate_accuracy(X, Y, parameters)
    # evaluate_accuracy(X_test, Y_test, parameters)
    # test_hidden_layer_size([1, 2, 3, 4, 5, 20, 50])
