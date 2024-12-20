import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков
import numpy as np  # Импортируем библиотеку для работы с массивами и математическими функциями
from math import e, log  # Импортируем константу e и функцию логарифма

# Данные для двух классов
X1 = [  # Координаты первого класса
    [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06],
    [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
]
X2 = [  # Координаты второго класса
    [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88],
    [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]
]

# Создаем входные данные и целевые метки
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))]  # Входные данные для первого класса
targets = [0 for _ in range(len(X1[0]))]  # Целевые метки для первого класса
inputs += [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))]  # Входные данные для второго класса
targets += [1 for _ in range(len(X2[0]))]  # Целевые метки для второго класса

# Инициализация весов с использованием NumPy
weights = np.random.randn(3) * 0.01  # Случайная инициализация весов

# Функция для вычисления взвешенного значения z
def weighted_z(point):
    z = [item * weights[i] for i, item in enumerate(point)]  # Умножаем каждую координату на соответствующий вес
    return sum(z) + weights[-1]  # Возвращаем сумму и добавляем смещение

# Функция логистической функции
def logistic_function(z):
    return 1 / (1 + e ** (-z))  # Возвращаем значение логистической функции

# Функция для вычисления ошибки логистической регрессии с регуляризацией
def logistic_error(reg_lambda=0.01):  # Добавлена регуляризация
    errors = []  # Список для хранения ошибок
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевую метку
        if output == 1:  # Если выходное значение равно 1
            output = 0.99999  # Избегаем логарифма от 0
        if output == 0:  # Если выходное значение равно 0
            output = 0.00001  # Избегаем логарифма от 0
        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))  # Вычисляем ошибку
        errors.append(error)  # Добавляем ошибку в список
    # Добавлена L2 регуляризация
    reg_term = (reg_lambda / 2) * np.sum(weights**2)  # Вычисляем регуляризационный член
    return (sum(errors) / len(errors)) + reg_term  # Возвращаем среднюю ошибку с регуляризацией

# Различные значения скорости обучения для эксперимента
lr_values = [0.01, 0.1, 0.2, 0.3]  
best_lr = 0.1  # Начальное значение для лучшей скорости обучения
min_error = float('inf')  # Начальное значение для минимальной ошибки

num_epochs = 10000  # Количество эпох для обучения

for lr in lr_values:  # Проходим по всем значениям скорости обучения
    weights = np.random.randn(3) * 0.01  # Переинициализация весов для каждого lr
    for epoch in range(num_epochs):  # Проходим по всем эпохам
        for i, point in enumerate(inputs):  # Проходим по всем входным данным
            z = weighted_z(point)  # Вычисляем z
            output = logistic_function(z)  # Получаем выходное значение
            target = targets[i]  # Получаем целевую метку
            for j in range(len(weights) - 1):  # Обновляем веса для всех, кроме смещения
                weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))  # Обновляем вес
            weights[-1] -= lr * (output - target) * (1 / len(inputs))  # Обновляем смещение
        error = logistic_error()  # Вычисляем ошибку
        if error < min_error:  # Если ошибка меньше минимальной
            min_error = error  # Обновляем минимальную ошибку
            best_lr = lr  # Обновляем лучшую скорость обучения

    print(f"lr: {lr}, error: {error}")  # Выводим скорость обучения и ошибку

print(f"Лучшая скорость обучения: {best_lr}")  # Выводим лучшую скорость обучения
print(weights)  # Выводим финальные веса

# Остальной код (accuracy, test, plot_decision_boundary) остается тем же

# Функция для вычисления точности модели
def accuracy():
    true_outputs = 0  # Счетчик правильных ответов
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевую метку
        if round(output) == target:  # Если округленное значение совпадает с целевой меткой
            true_outputs += 1  # Увеличиваем счетчик
    return true_outputs, len(inputs)  # Возвращаем количество правильных ответов и общее количество

# Функция для тестирования модели
def test():
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевую метку
        print(f"output: {round(output, 2)}, target: {target}")  # Выводим выходное значение и целевую метку

test()  # Запускаем тестирование
print("accuracy:", accuracy())  # Выводим точность

# Функция для построения границы принятия решения
def plot_decision_boundary():
    x_values = [i / 10 for i in range(30)]  # Генерируем значения x
    y_values = [-(weights[0] * x + weights[-1]) / weights[1] for x in x_values]  # Вычисляем соответствующие значения y
    plt.plot(x_values, y_values, color='red', label='Decision Boundary')  # Строим границу принятия решения

plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
plt.scatter(X1[0], X1[1], color='purple', label='Class 0 (X1)')  # Строим точки первого класса
plt.scatter(X2[0], X2[1], color='violet', label='Class 1 (X2)')  # Строим точки второго класса
plot_decision_boundary()  # Строим границу принятия решения
plt.title('Logistic regression')  # Устанавливаем заголовок графика
plt.xlabel('X values')  # Устанавливаем подпись для оси X
plt.ylabel('Y values')  # Устанавливаем подпись для оси Y
plt.legend()  # Показываем легенду
plt.grid()  # Включаем сетку
plt.show()  # Отображаем график
