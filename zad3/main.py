import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
print("\nZESTAW 1 ZADANIE 1 i 2 \n")

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
print("\nZESTAW 1 ZADANIE 3 \n")

X = np.array([[2000], [2002], [2005], [2007], [2010]])
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




# 2.3 Zadania do zrobienia

# ZADANIE1a
# Napisz Perceptron-Learn Program, który znajdzie odpowiednie perceptrony
# reprezentujące funkcje x1 ∧ x2 czyli funkcja AND
print("\nZESTAW 2 ZADANIE 1a \n")

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
print("\nZESTAW 2 ZADANIE 1b \n")

X = np.array([[0], [1]])
d = np.array([1,0])

perceptron = Perceptron(input_size=1)
perceptron.fit(X, d)

print(perceptron.W)

#ZADANIE 2
# Zaprojektuj perceptron z dwoma wejściami reprezentujący funkcję boolowską
# x1 ∧ ¬x2.
print("\nZESTAW 2 ZADANIE 2 \n")

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
print("\nZESTAW 2 ZADANIE 3 \n")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(np.round(nn.output))


# ZADANIE 4
# Napisz algorytm propagacji wstecznej do reprezentacji x1 XOR x2.
print("\nZESTAW 2 ZADANIE 4 \n")

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
# Biorąc pod uwagę następującą sieć neuronową z zainicjowanymi wagami, jak
# na rysunku, wyjaśnij sieć architektura. Wynik sieci powinien być w stanie
# poprawnie sklasyfikować wyniki tam, gdzie na rysunku podano przykłady
# treningowe. Niech współczynnik uczenia się α będzie równy 0.1, a wagi będą
# takie, jak pokazano na poniższym rysunku. Zrób przód propagację sygnałów
# w sieci przy użyciu pierwszy przykład jako wejścia, a następnie wykonać
# propagację wsteczną błąd. Pokaż zmiany wag.
print("\nZESTAW 2 ZADANIE 5 \n")

input_layer_size = 2
hidden_layer_size = 3
output_layer_size = 2

weights1 = np.array([
    [0.1, -0.2],
    [0, 0.2],
    [0.3, -0.4]
])

weights2 = np.array([
    [-0.4, 0.1, 0.6],
    [-0.1, -0.2, 0.2]
])

training_inputs = np.array([[0.2, 0.3], [0.4, 0.5], [0.1, 0.7]])
training_outputs = np.array([[0, 1], [1, 0], [0, 1]])

learning_rate = 0.1
for i in range(10000):
    # propagacja w przód
    hidden_layer_outputs = sigmoid(np.dot(training_inputs, weights1.T))
    output_layer_outputs = sigmoid(np.dot(hidden_layer_outputs, weights2.T))

    # propagacja wsteczna błędu
    output_layer_errors = (training_outputs - output_layer_outputs) * output_layer_outputs * (1 - output_layer_outputs)
    hidden_layer_errors = np.dot(output_layer_errors, weights2) * hidden_layer_outputs * (1 - hidden_layer_outputs)

    # aktualizacja wag
    weights2 += learning_rate * np.dot(output_layer_errors.T, hidden_layer_outputs)
    weights1 += learning_rate * np.dot(hidden_layer_errors.T, training_inputs)

test_input = np.array([[0.6,0.1],[0.2,0.3]])
hidden_layer_output = sigmoid(np.dot(test_input, weights1.T))
output_layer_output = sigmoid(np.dot(hidden_layer_output, weights2.T))


print("Nowe wagi dla warstwy ukrytej:")
print(weights1,"\n")
print("Nowe wagi dla warstwy wyjściowej:")
print(weights2,"\n")

print("Wynik dla testowego wejścia", test_input, "wynosi:", np.round(output_layer_output[0]))
