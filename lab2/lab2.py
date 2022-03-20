import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def Relu(x):
    return x * (x > 0)


def derivative_Relu(x):
    return 1. * (x > 0)


def MSE(y, y_pred):
    '''
    y: grouth true
    y_pred: pred
    '''
    square_list = []
    for i in range(y.size):
        s = (y[i] - y_pred[i]) * (y[i] - y_pred[i])
        square_list.append(s)

    total = 0.0
    for j in range(len(square_list)):
        total += square_list[j]

    return total / (len(square_list))


def derivative_MSE(y, y_pred):
    return -2 * (y - y_pred) / y.shape[0]


class NNet():
    def __init__(self, structure):
        self.layers = structure
        self.w1 = np.random.normal(0.0, 1.0, (self.layers[0], self.layers[1]))
        self.w2 = np.random.normal(0.0, 1.0, (self.layers[1], self.layers[2]))
        self.w3 = np.random.normal(0.0, 1.0, (self.layers[2], self.layers[3]))

        self.v1 = np.random.normal(0.0, 1.0, (self.layers[0], self.layers[1]))
        self.v2 = np.random.normal(0.0, 1.0, (self.layers[1], self.layers[2]))
        self.v3 = np.random.normal(0.0, 1.0, (self.layers[2], self.layers[3]))

        self.m1 = np.random.normal(0.0, 1.0, (self.layers[0], self.layers[1]))
        self.m2 = np.random.normal(0.0, 1.0, (self.layers[1], self.layers[2]))
        self.m3 = np.random.normal(0.0, 1.0, (self.layers[2], self.layers[3]))

        self.b = 0.01

    def forward(self, x):
        self.x = x
        self.z1 = (self.x @ self.w1) + self.b
        if args.a == 'sigmoid':
            self.a1 = sigmoid(self.z1)
            self.z2 = (self.a1 @ self.w2) + self.b
            self.a2 = sigmoid(self.z2)
            self.z3 = (self.a2 @ self.w3) + self.b
            self.y_pred = sigmoid(self.z3)
        elif args.a == 'relu':
            self.a1 = Relu(self.z1)
            self.z2 = (self.a1 @ self.w2) + self.b
            self.a2 = Relu(self.z2)
            self.z3 = (self.a2 @ self.w3) + self.b
            self.y_pred = Relu(self.z3)
        return self.y_pred

    def backprop(self, y, y_pred):
        d_loss = derivative_MSE(y, y_pred)
        if args.a == 'sigmoid':
            d_y_pred = derivative_sigmoid(self.y_pred)
            d_a2 = derivative_sigmoid(self.a2)
            d_a1 = derivative_sigmoid(self.a1)
        elif args.a == 'relu':
            d_y_pred = derivative_Relu(self.y_pred)
            d_a2 = derivative_Relu(self.a2)
            d_a1 = derivative_Relu(self.a1)

        self.d_L_w3 = self.a2.T @ (d_y_pred * d_loss)
        self.d_L_w2 = self.a1.T @ (d_a2 * ((d_y_pred * d_loss) @ self.w3.T))
        self.d_L_w1 = self.x.T @ (d_a1 * ((d_a2 * ((d_y_pred * d_loss) @ self.w3.T)) @ self.w2.T))


    def optimize(self, lr):
        if args.o == 'sgd':
            self.w1 = self.w1 - lr * self.d_L_w1
            self.w2 = self.w2 - lr * self.d_L_w2
            self.w3 = self.w3 - lr * self.d_L_w3
        elif args.o == 'momentum':
            self.v1 = 0.9 * self.v1 - lr * self.d_L_w1
            self.v2 = 0.9 * self.v2 - lr * self.d_L_w2
            self.v3 = 0.9 * self.v3 - lr * self.d_L_w3

            self.w1 = self.w1 + self.v1
            self.w2 = self.w2 + self.v2
            self.w3 = self.w3 + self.v3


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Choose the dataset: (0) linear, (1) XOR", type=int, default=0)
    parser.add_argument("-a", help="Choose the activation function: (sigmoid), (relu)", type=str, default="sigmoid")
    parser.add_argument("-o", help="Choose the optimizer: (sgd), (momentum), (adam)", type=str, default="sgd")
    args = parser.parse_args()

    np.random.seed(0)
    lr = 0.1
    epochs = 10000
    structure = [2, 4, 4, 1]
    loss_list = []
    acc_list = []

    if args.d == 0:
        x, y = generate_linear(n=100)
    else:
        x, y = generate_XOR_easy()

    model = NNet(structure)

    for epoch in range(1, epochs + 1):
        y_pred = model.forward(x)
        loss = MSE(y, y_pred)
        loss_list.append(loss)
        model.backprop(y, y_pred)
        model.optimize(lr)
        acc = np.count_nonzero(np.round(y_pred) == y) * 100 / len(y_pred)
        acc_list.append(acc / 100)
        if (epoch) % 500 == 0:
            print(f'epoch {epoch} loss : {loss[0]}')


    y_pred = model.forward(x)
    print(y_pred)
    print('Epoch: ', epochs)
    print('Learning Rate: ', lr)
    print('Test Accuracy (%): {:3.2f}%'.format(np.count_nonzero(np.round(y_pred) == y) * 100 / len(y_pred)))


    _e = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.title('Linear loss / acc (.01%) Fig.')
    plt.plot(_e, loss_list, label='training loss')
    plt.plot(_e, acc_list, label='accuracy')
    plt.xlabel('epoch iters')
    plt.ylabel('loss / acc (.01%)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()