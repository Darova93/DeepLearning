import matplotlib.pyplot as plt
from PIL import Image
from dnn_app_utils_v3 import *
from parameter_initializer import InitType, ParameterInitializer
from cost_calculator import CostType, Cost
from forward_propagation import FwdPropType, ForwardPropagation
from backward_propagation import BwdPropType, BackwardPropagation


def update_parameter(initial_parameters, grads, learning_rate):
    total_layers = len(initial_parameters) // 2

    for i in range(total_layers):
        initial_parameters["W"+str(i+1)] = initial_parameters["W"+str(i+1)] - learning_rate*grads["dW"+str(i+1)]
        initial_parameters["b"+str(i+1)] = initial_parameters["b"+str(i+1)] - learning_rate*grads["db"+str(i+1)]

    return initial_parameters


def display_image(index):
    train_x_orig, train_y_data, test_x_orig, test_y_data, classes_data = load_data()
    plt.imshow(train_x_orig[index])
    plt.show()
    print("y = "+str(train_y_data[0, index])+". It's a " + classes[train_y_data[0, index]].decode("utf-8")+" picture.")


def nn_model(x, y, layers_dimensions, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    np.random.seed(1)
    parameters = ParameterInitializer(layers_dimensions, InitType.Xavier).initialize()

    for i in range(0, num_iterations):
        al, caches = ForwardPropagation(FwdPropType.Backdrop, keep_prob=0.86).compute(x, parameters)
        cost = Cost(CostType.Default, hyp_lambda=0.7).compute(al, y, parameters)
        grads = BackwardPropagation(BwdPropType.Backdrop, hyp_lambda=0.7, keep_prob=0.86).compute(al, y, caches)
        parameters = update_parameter(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost.squeeze()))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(x, y, parameters):
    m = x.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = ForwardPropagation(FwdPropType.Default).compute(x, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(set_classes, x, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(x[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " +
            set_classes[int(p[0, index])].decode("utf-8")+" \n Class: " +
            set_classes[y[0, index]].decode("utf-8"))
    plt.show()


def evaluate_image(image_name, parameters, is_cat=True):
    image = np.array(Image.open(image_name).resize((64, 64))) / 255.
    flattened_image = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], -1)
    my_predicted_image = predict(flattened_image, [is_cat*1.], parameters)

    # plt.imshow(image)
    # plt.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # train_x, train_y, test_x, test_y, classes = load_data()  # 2500, 0.0075, (12288, 20, 7, 5, 1)
    train_x, train_y, test_x, test_y = load_2d_dataset()  # 30000, 0.3, (2, 20, 3, 1)
    layers_dims = (train_x.shape[0], 20, 3, 1)

    trained_parameters = nn_model(train_x,
                                  train_y,
                                  layers_dims,
                                  learning_rate=0.3,
                                  num_iterations=30000,
                                  print_cost=True)

    print('Train set accuracy')
    predictions_train = predict(train_x, train_y, trained_parameters)
    print('Dev set accuracy')
    predictions_test = predict(test_x, test_y, trained_parameters)

    # evaluate_image('datasets/cat.png', trained_parameters)
