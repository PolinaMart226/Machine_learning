 # Импортируем математические функции e и log
from math import e, log
# Импортируем функцию randint для генерации случайных чисел
from random import randint 
 # Импортируем библиотеку для построения графиков
import matplotlib.pyplot as plt

# Данные в формате матриц
# Данные для первого класса
X1 = [  
    [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06],  # x1
    [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]   # y1
]
 # Данные для второго класса
X2 = [ 
    [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88],  # x2
    [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]  # y2
]

# Объединяем данные
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))]  # Создаем список входных данных для класса 0
targets = [0 for _ in range(len(X1[0]))]  # Создаем список целевых значений для класса 0
inputs += [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))]  # Добавляем входные данные для класса 1
targets += [1 for _ in range(len(X2[0]))]  # Добавляем целевые значения для класса 1

weights = [randint(-100, 100) / 100 for _ in range(3)]  # Инициализируем веса случайными значениями

def weighted_z(point):  # Функция для вычисления взвешенной суммы
    z = [item * weights[i] for i, item in enumerate(point)]  # Умножаем каждую координату на соответствующий вес
    return sum(z) + weights[-1]  # Возвращаем сумму и добавляем смещение (bias)

def logistic_function(z):  # Логистическая функция
    return 1 / (1 + e ** (-z))  # Возвращаем значение логистической функции

def logistic_error():  # Функция для вычисления ошибки логистической регрессии
    errors = []  # Список для хранения ошибок
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем взвешенную сумму
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        if output == 1:  # Если выход равен 1, заменяем его на 0.99999 для избежания логарифма нуля
            output = 0.99999
        if output == 0:  # Если выход равен 0, заменяем его на 0.00001
            output = 0.00001

        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))  # Вычисляем ошибку
        errors.append(error)  # Добавляем ошибку в список

    return sum(errors) / len(errors)  # Возвращаем среднюю ошибку

# Гиперпараметры
lr = 0.1  # Скорость обучения
num_epochs = 100  # Количество эпох

# Обучение модели
for epoch in range(num_epochs):  # Проходим по всем эпохам
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем взвешенную сумму
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        for j in range(len(weights) - 1):  # Обновляем веса для входных данных
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))  # Обновляем вес

        weights[-1] -= lr * (output - target) * (1 / len(inputs))  # Обновляем смещение (bias)

    # Вывод значений epoch, error, x1, x2, bias, output и target
    error = logistic_error()  # Вычисляем ошибку
    print(f"epoch: {epoch}, error: {error}, x1: {inputs[i][0]}, x2: {inputs[i][1]}, bias: {weights[-1]}, output: {round(output, 2)}, target: {target}")  # Выводим информацию об эпохе

print(weights)  # Выводим финальные веса

def accuracy():  # Функция для вычисления точности модели
    true_outputs = 0  # Счетчик правильных ответов
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем взвешенную сумму
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        if round(output) == target:  # Если округленное выходное значение совпадает с целевым
            true_outputs += 1  # Увеличиваем счетчик правильных ответов

    return true_outputs, len(inputs)  # Возвращаем количество правильных ответов и общее количество данных

def test():  # Функция для тестирования модели
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем взвешенную сумму
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение
        print(f"output: {round(output, 2)}, target: {target}")  # Выводим выходное значение и целевое значение

test()  # Запускаем тестирование
print("accuracy:", accuracy())  # Выводим точность модели

# Графическая интерпретация
plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
plt.scatter(X1[0], X1[1], color='purple', label='Class 0 (X1)')  # Отображаем точки первого класса
plt.scatter(X2[0], X2[1], color='violet', label='Class 1 (X2)')  # Отображаем точки второго класса
plt.title('Logistic regression')  # Заголовок графика
plt.xlabel('X values')  # Подпись оси X
plt.ylabel('Y values')  # Подпись оси Y
plt.legend()  # Отображаем легенду
plt.grid()  # Включаем сетку
plt.show()  # Показываем график
