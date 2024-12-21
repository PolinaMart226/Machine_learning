# Импортируем библиотеку для построения графиков
import matplotlib.pyplot as plt
# Импортируем библиотеку для работы с массивами и математическими функциями
import numpy as np
# Импортируем константу e и функцию логарифма
from math import e, log

# Данные для двух классов
# Координаты первого класса
X1 = [ 
    [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06],
    [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
]
# Координаты второго класса
X2 = [ 
    [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88],
    [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]
]

# Создаем входные данные и целевые метки
# Входные данные для первого класса
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))]
 # Целевые метки для первого класса
targets = [0 for _ in range(len(X1[0]))]
# Входные данные для второго класса
inputs += [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))]
# Целевые метки для второго класса
targets += [1 for _ in range(len(X2[0]))] 

# Инициализация весов с использованием NumPy
# Случайная инициализация весов
weights = np.random.randn(3) * 0.01

# Функция для вычисления взвешенного значения z
def weighted_z(point):
    # Умножаем каждую координату на соответствующий вес
    z = [item * weights[i] for i, item in enumerate(point)]
    # Возвращаем сумму и добавляем смещение
    return sum(z) + weights[-1]

# Функция логистической функции
def logistic_function(z):
    # Возвращаем значение логистической функции
    return 1 / (1 + e ** (-z))  

# Функция для вычисления ошибки логистической регрессии с регуляризацией
# Добавлена регуляризация
def logistic_error(reg_lambda=0.01):
    # Список для хранения ошибок
    errors = []
    # Проходим по всем входным данным
    for i, point in enumerate(inputs):
        # Вычисляем z
        z = weighted_z(point)
        # Получаем выходное значение
        output = logistic_function(z)
        # Получаем целевую метку
        target = targets[i]
        # Если выходное значение равно 1
        if output == 1:
            # Избегаем логарифма от 0
            output = 0.99999
            # Если выходное значение равно 0
        if output == 0:
            # Избегаем логарифма от 0
            output = 0.00001
            # Вычисляем ошибку
        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
         # Добавляем ошибку в список
        errors.append(error)
    # Добавлена L2 регуляризация
    # Вычисляем регуляризационный член
    reg_term = (reg_lambda / 2) * np.sum(weights**2)
    # Возвращаем среднюю ошибку с регуляризацией
    return (sum(errors) / len(errors)) + reg_term

# Различные значения скорости обучения для эксперимента
lr_values = [0.01, 0.1, 0.2, 0.3]
# Начальное значение для лучшей скорости обучения
best_lr = 0.1
# Начальное значение для минимальной ошибки
min_error = float('inf')

# Количество эпох для обучения
num_epochs = 10000 

# Проходим по всем значениям скорости обучения
for lr in lr_values:
    # Переинициализация весов для каждого lr
    weights = np.random.randn(3) * 0.01
    # Проходим по всем эпохам
    for epoch in range(num_epochs):  
        # Проходим по всем входным данным
        for i, point in enumerate(inputs):
            # Вычисляем z
            z = weighted_z(point)
            # Получаем выходное значение
            output = logistic_function(z)
            # Получаем целевую метку
            target = targets[i]
            # Обновляем веса для всех, кроме смещения
            for j in range(len(weights) - 1):
                # Обновляем вес
                weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))
                # Обновляем смещение
            weights[-1] -= lr * (output - target) * (1 / len(inputs))
             # Вычисляем ошибку
        error = logistic_error()
        # Если ошибка меньше минимальной
        if error < min_error:
            # Обновляем минимальную ошибку
            min_error = error
            # Обновляем лучшую скорость обучения
            best_lr = lr 

# Выводим скорость обучения и ошибку
print(f"lr: {lr}, error: {error}") 
# Выводим лучшую скорость обучения
print(f"Лучшая скорость обучения: {best_lr}")
# Выводим финальные веса
print(weights) 

# Функция для вычисления точности модели
def accuracy():
    # Счетчик правильных ответов
    true_outputs = 0
    # Проходим по всем входным данным
    for i, point in enumerate(inputs):
        # Вычисляем z
        z = weighted_z(point)
        # Получаем выходное значение
        output = logistic_function(z)
        # Получаем целевую метку
        target = targets[i]
        # Если округленное значение совпадает с целевой меткой
        if round(output) == target:
            # Увеличиваем счетчик
            true_outputs += 1
            # Возвращаем количество правильных ответов и общее количество
    return true_outputs, len(inputs)  

# Функция для тестирования модели
def test():
    # Проходим по всем входным данным
    for i, point in enumerate(inputs):
        # Вычисляем z
        z = weighted_z(point)
        # Получаем выходное значение
        output = logistic_function(z)
        # Получаем целевую метку
        target = targets[i]
        # Выводим выходное значение и целевую метку
        print(f"output: {round(output, 2)}, target: {target}")  

# Запускаем тестирование
test()
# Выводим точность
print("accuracy:", accuracy())  

# Функция для построения границы принятия решения
def plot_decision_boundary():
    # Генерируем значения x
    x_values = [i / 10 for i in range(30)]
    # Вычисляем соответствующие значения y
    y_values = [-(weights[0] * x + weights[-1]) / weights[1] for x in x_values]
    # Строим границу принятия решения
    plt.plot(x_values, y_values, color='red', label='Decision Boundary') 

# Устанавливаем размер графика
plt.figure(figsize=(10, 6))
# Строим точки первого класса
plt.scatter(X1[0], X1[1], color='purple', label='Class 0 (X1)')
# Строим точки второго класса
plt.scatter(X2[0], X2[1], color='violet', label='Class 1 (X2)')
plot_decision_boundary()  # Строим границу принятия решения
plt.title('Logistic regression')  # Устанавливаем заголовок графика
plt.xlabel('X values')  # Устанавливаем подпись для оси X
plt.ylabel('Y values')  # Устанавливаем подпись для оси Y
plt.legend()  # Показываем легенду
plt.grid()  # Включаем сетку
plt.show()  # Отображаем график
