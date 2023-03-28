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








