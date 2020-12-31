import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
from PIL import Image


def show_image(image):
    plt.imshow(image)
    plt.show()


def show_cost(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def test_training_rates(learning_rates, data_sets):
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(data_sets, num_iterations=1500, learning_rate=i, print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def predict_custom_image(image_name, r, c, print_image=False):
    my_image = image_name
    image = Image.open(my_image)
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.
    flattened_image = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    my_predicted_image = predict(r["w"], r["b"], flattened_image)

    if print_image:
        show_image(image)

    print(
        "y = " +
        str(np.squeeze(my_predicted_image)) +
        ", your algorithm predicts a \"" +
        c[int(np.squeeze(my_predicted_image)), ].decode("utf-8") +
        "\" picture.")


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, x, y):
    m = x.shape[1]
    a = sigmoid(np.dot(w.T, x) + b)
    cost = (-1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    dw = (1 / m) * (np.dot(x, (a - y).T))
    db = (1 / m) * np.sum(a - y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = []
    db = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, x, y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost.squeeze()))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, x):
    m = x.shape[1]
    w = w.reshape(x.shape[0], 1)
    a = sigmoid(np.dot(w.T, x) + b)
    y_prediction = np.where(a > 0.5, 1, 0)
    assert (y_prediction.shape == (1, m))
    return y_prediction


def model(data_sets, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(data_sets["train_set_x"].shape[0])
    parameters, grads, costs = optimize(
        w,
        b,
        data_sets["train_set_x"],
        data_sets["train_set_y"],
        num_iterations,
        learning_rate,
        print_cost)
    w = parameters["w"]
    b = parameters["b"]
    y_prediction_train = predict(w, b, data_sets["train_set_x"])
    y_prediction_test = predict(w, b, data_sets["test_set_x"])
    y_train_accuracy = (100 - np.mean(np.abs(y_prediction_train - data_sets["train_set_y"])) * 100)
    y_test_accuracy = (100 - np.mean(np.abs(y_prediction_test - data_sets["test_set_y"])) * 100)
    print("train accuracy: {} %".format(y_train_accuracy))
    print("test accuracy: {} %".format(y_test_accuracy))
    if print_cost:
        show_cost(costs, learning_rate)

    return {
        "costs": costs,
        "Y_prediction_test": y_prediction_test,
        "Y_prediction_test_accuracy": y_test_accuracy,
        "Y_prediction_train": y_prediction_train,
        "Y_prediction_train_accuracy": y_train_accuracy,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }


def load_sets():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, loaded_classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    
    return {
        "train_set_x": train_set_x,
        "train_set_y": train_set_y,
        "test_set_x": test_set_x,
        "test_set_y": test_set_y,
        "classes": loaded_classes,
    }


if __name__ == '__main__':
    iterations = 2000
    training_rate = 0.0015
    
    sets = load_sets()
    result = model(sets, iterations, training_rate, False)

    # test_training_rates([0.01, 0.001, 0.0001], sets)
    # predict_custom_image('cat.png', model, sets["classes"], True)
