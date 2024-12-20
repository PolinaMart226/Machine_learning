from math import e, log
from random import randint
import matplotlib.pyplot as plt  # type: ignore # Импортируем библиотеку для графиков

# Данные в формате матриц
X1 = [  # Данные для первого класса
    [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06],  # x1
    [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]   # y1
]
X2 = [  # Данные для второго класса
    [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88],  # x2
    [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]  # y2
]

# Объединяем данные
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))]
targets = [0 for _ in range(len(X1[0]))]
inputs += [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))]
targets += [1 for _ in range(len(X2[0]))]

weights = [randint(-100, 100) / 100 for _ in range(3)]

def weighted_z(point):
    z = [item * weights[i] for i, item in enumerate(point)]
    return sum(z) + weights[-1]

def logistic_function(z):
    return 1 / (1 + e ** (-z))

def logistic_error():
    errors = []
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        if output == 1:
            output = 0.99999
        if output == 0:
            output = 0.00001

        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
        errors.append(error)

    return sum(errors) / len(errors)

# Гиперпараметры
lr = 0.1  # Скорость обучения
num_epochs = 100  # Количество эпох

# Обучение модели
for epoch in range(num_epochs):
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        # Вывод значений epoch, error, x1, x2, bias, output, target
        bias = weights[-1]
        print(f"epoch: {epoch}, error: {logistic_error()}, x1: {point[0]}, x2: {point[1]}, bias: {bias}, output: {output}, target: {target}")

        for j in range(len(weights) - 1):
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))

        weights[-1] -= lr * (output - target) * (1 / len(inputs))
        
    print(f"epoch: {epoch}, error: {logistic_error()}")

print(weights)

def accuracy():
    true_outputs = 0
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]

        if round(output) == target:
            true_outputs += 1

    return true_outputs, len(inputs)

def test():
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        print(f"output: {round(output, 2)}, target: {target}")

test()
print("accuracy:", accuracy())

# Графическая интерпретация
plt.figure(figsize=(10, 6))
plt.scatter(X1[0], X1[1], color='blue', label='Class 0 (X1)')
plt.scatter(X2[0], X2[1], color='red', label='Class 1 (X2)')

# Добавление линии пересечения
x_values = [min(X1[0] + X2[0]), max(X1[0] + X2[0])]
y_values = [(-weights[0] * x - weights[-1]) / weights[1] for x in x_values]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.title('Logistic regression')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid()
plt.show()
