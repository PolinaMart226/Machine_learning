import numpy as np
import matplotlib.pyplot as plt # type: ignore

from math import e, log
from random import randint

# Данные для x1 и y1
x1 = np.array([[0.38, 1.79], [1.42, 0.54], [0.55, 0.34], [1.34, 0.678], 
                [1.76, 1.64], [1.62, 0.92], [0.83, 1.49], [0.84, 0.3], 
                [1.77, 0.7], [1.06, 0.99]])

# Данные для x2 и y2
x2 = np.array([[3.9, 4.93], [6.14, 4.95], [6.1, 0.97], [2.11, 0.77], 
                [3.23, 0.43], [1.62, 4.61], [1.88, 0.25]])

# Объединение данных
inputs = np.vstack((x1, x2))
targets = np.array([0] * len(x1) + [1] * len(x2))

# Инициализация весов
weights = np.random.uniform(-1, 1, size=(inputs.shape[1] + 1))  # 2 веса + 1 смещение

def weighted_z(point):
    z = np.dot(point, weights[:-1]) + weights[-1]  # Векторное произведение
    return z

def logistic_function(z):
    return 1 / (1 + e ** (-z))

def logistic_error():
    errors = []
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        # Избегаем логарифма от 0
        output = max(min(output, 0.99999), 0.00001)

        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
        errors.append(error)

    return sum(errors) / len(errors)

# Гиперпараметры
lr = 0.01  # Скорость обучения
epochs = 200  # Количество эпох

# Обучение модели
for epoch in range(epochs):
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        # Обновление весов
        weights[:-1] -= lr * point * (output - target) / len(inputs)
        weights[-1] -= lr * (output - target) / len(inputs)

    if epoch % 20 == 0:  # Печатаем каждые 20 эпох
        print(f"epoch: {epoch}, error: {logistic_error()}")

# Оценка качества модели
def accuracy():
    true_outputs = 0
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        if round(output) == target:
            true_outputs += 1
    return true_outputs, len(inputs)

# Графическая интерпретация
def plot_decision_boundary():
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    Z = np.array([logistic_function(weighted_z(np.array([x, y]))) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap='RdYlBu')
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, edgecolors='k', cmap='RdYlBu')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Вызов функции для построения графика
plot_decision_boundary()

# Печать весов и точности
print("Final weights:", weights)
print("Accuracy:", accuracy())