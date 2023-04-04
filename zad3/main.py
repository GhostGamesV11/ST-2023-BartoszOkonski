import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 2.3 Zadania do zrobienia

# ZADANIE1a
# Napisz Perceptron-Learn Program, który znajdzie odpowiednie perceptrony
# reprezentujące funkcje x1 ∧ x2 czyli funkcja AND
#
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0,0,0,1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print(perceptron.W)

#ZADANIE1b
# Napisz Perceptron-Learn Program, który znajdzie odpowiednie perceptrony
# reprezentujące funkcje ¬x1 czyli funkcje NOT
#
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0], [1]])
d = np.array([1,0])

perceptron = Perceptron(input_size=1)
perceptron.fit(X, d)

print(perceptron.W) 

#ZADANIE2
# Zaprojektuj perceptron z dwoma wejściami reprezentujący funkcję boolowską
# x1 ∧ ¬x2.
#
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[1,0], [1,1], [0,1], [0,0]])
d = np.array([1,0,0,0])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print(perceptron.W)

# ZADANIE 3
# Zaprojektuj dwuwarstwowa sieć perceptronów implementująca x1 XOR x2.
#
class Perceptron:
    def __init__(self, input_size, output_size, lr=1, epochs=10):
        self.W = np.random.rand(input_size+1, output_size) * 2 - 1
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                delta = e * self.activation_derivative(y)
                self.W = self.W + self.lr * np.outer(x, delta)

class XORPerceptron:
    def __init__(self):
        self.hidden_layer = Perceptron(input_size=2, output_size=2)
        self.output_layer = Perceptron(input_size=2, output_size=1)

    def predict(self, x):
        h = self.hidden_layer.predict(x)
        o = self.output_layer.predict(h)
        return np.round(o)

    def fit(self, X, d):
        for epoch in range(self.hidden_layer.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                h = self.hidden_layer.predict(x)
                o = self.output_layer.predict(h)
                e = d[i] - o
                delta = e * self.output_layer.activation_derivative(o)
                delta_h = self.hidden_layer.activation_derivative(h) * delta.dot(self.output_layer.W.T)
                x = np.insert(x, 0, 1)
                self.output_layer.W = self.output_layer.W + self.output_layer.lr * np.outer(h, delta)
                self.hidden_layer.W = self.hidden_layer.W + self.hidden_layer.lr * np.outer(x, delta_h)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0, 1, 1, 0])

perceptron = XORPerceptron()
perceptron.fit(X, d)

print(perceptron.predict(X))

# ZADANIE 4
# Napisz algorytm propagacji wstecznej do reprezentacji x1 XOR x2.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
synapse_0 = 2 * np.random.random((2, 4)) - 1
synapse_1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    layer_2_error = y - layer_2

    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    synapse_1 += layer_1.T.dot(layer_2_delta)
    synapse_0 += layer_0.T.dot(layer_1_delta)

print("Wynik po nauczeniu:")
print(layer_2)

# ZADANIE 5
# Definicja funkcji aktywacji sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Definicja pochodnej funkcji aktywacji sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Definicja klasy sieci neuronowej
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicjalizacja wag dla warstwy ukrytej i wyjściowej
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_output = np.random.randn(self.hidden_size, self.output_size)
        # self.weights_hidden = [[0.1,-0.2,0.1],[0.2,0.2,0],[-0.4,0.5,0.3]]
        # self.weights_output = [[0.1,-0.4],[0.2,-0.1],[-0.2,0.6]]

        print(self.weights_hidden)
        print(self.weights_output)

    def forward(self, X):
        # Propagacja do przodu
        self.hidden_layer = sigmoid(np.dot(X, self.weights_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_output))
        return self.output_layer

    def backward(self, X, y, learning_rate):
        # Propagacja wsteczna
        error = y - self.output_layer
        d_output = error * sigmoid_derivative(self.output_layer)

        error_hidden = np.dot(d_output, self.weights_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_layer)

        # Aktualizacja wag
        self.weights_output += learning_rate * np.dot(self.hidden_layer.T, d_output)
        self.weights_hidden += learning_rate * np.dot(X.T, d_hidden)

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return np.round(self.forward(X))

# Utworzenie instancji klasy sieci neuronowej
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Dane uczące i docelowe wyjście
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Uczenie sieci neuronowej
nn.train(X, y, epochs=15000, learning_rate=0.1)

# Testowanie sieci neuronowej
test_data = np.array([[0.6,0.1],[0.2,0.3]])
predictions = nn.predict(test_data)
print(predictions)



# 1.3 Zadania do zrobienia

#ZADANIE1 i ZADANIE2
# Rozważ poniższe dane, które pokazują procent bezrobotnych w wieku 25 lat
# (lub starszych) w mieście w pewnym okresie czasu, podanym w latach.
#
# Year 2000 2002 2005 2007 2010
# Percentage 6.5 7.0 7.4 8.2 9.0
#
# Napisz program, który znajdzie model regresji liniowej do przewidywania
# procentu bezrobotnych w danym roku z dokładnością do trzech miejsc po
# przecinku.

# Korzystając z otrzymanego modelu określ, w którym roku procent ten przekroczy
# 12%.



X = np.array([[2000], [2002], [2005], [2007], [2010]])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])


model = LinearRegression()

model.fit(X, y)

rok = [[2023]]
przewidywany_procent = model.predict(rok)

print("Przewidywany procent bezrobotnych w 2021 roku: {:.3f}%".format(przewidywany_procent[0]))

# ZADANIE 3
# Przedstawić proces znajdowania regresji liniowej przy użyciu niektórych tech-
# nik animacji z biblioteki (e.g. matplotlib.pyplot) Pythona.


X = X = np.array([[2000], [2002], [2005], [2007], [2010]])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])


model = LinearRegression()


fig, ax = plt.subplots()
ax.scatter(X, y)


line, = ax.plot([], [])


def animate(i):

    model.fit(X, y)
    a = model.coef_[0]
    b = model.intercept_

    y_pred = a * X + b

    line.set_data(X, y_pred)

    return line,


ani = animation.FuncAnimation(fig, animate, frames=10, blit=True)
plt.show()








