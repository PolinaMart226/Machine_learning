from math import e, log  # Импортируем константу e и функцию логарифма
from random import randint  # Импортируем функцию randint для генерации случайных чисел

# Данные для первой группы
x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]
y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
# Данные для второй группы
x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]
y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]

# Объединяем входные данные в кортежи
inputs = [(x1[i], y1[i]) for i in range(len(x1))]
# Создаем целевые значения для первой группы (0)
targets = [0 for i in range(len(x1))]
# Добавляем данные второй группы
inputs += [(x2[i], y2[i]) for i in range(len(x2))]
# Создаем целевые значения для второй группы (1)
targets += [1 for i in range(len(x2))]

# Инициализируем веса случайными значениями
weights = [randint(-100, 100) / 100 for _ in range(3)]

# Функция для вычисления взвешенного значения z
def weighted_z(point):
    z = [item * weights[i] for i, item in enumerate(point)]  # Умножаем каждую координату на соответствующий вес
    return sum(z) + weights[-1]  # Возвращаем сумму и добавляем смещение (последний вес)

# Логистическая функция
def logistic_function(z):
    return 1 / (1 + e ** (-z))  # Возвращаем значение логистической функции

# Функция для вычисления ошибки логистической регрессии
def logistic_error():
    errors = []  # Список для хранения ошибок

    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        # Обработка крайних случаев для выхода
        if output == 1:
            output = 0.99999  # Избегаем деления на ноль
        if output == 0:
            output = 0.00001  # Избегаем деления на ноль

        # Вычисляем ошибку
        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
        errors.append(error)  # Добавляем ошибку в список

    return sum(errors) / len(errors)  # Возвращаем среднюю ошибку

lr = 0.1  # Устанавливаем скорость обучения

# Основной цикл обучения
for epoch in range(100):  # Проходим по эпохам
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        # Обновляем веса
        for j in range(len(weights) - 1):
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))  # Обновляем веса для входных данных

        weights[-1] -= lr * (output - target) * (1 / len(inputs))  # Обновляем вес смещения

    print(f"epoch: {epoch}, error: {logistic_error()}")  # Выводим номер эпохи и текущую ошибку

print(weights)  # Выводим финальные веса

# Функция для вычисления точности модели
def accuracy():
    true_outputs = 0  # Счетчик правильных ответов

    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение

        if round(output) == target:  # Проверяем, совпадает ли предсказание с целевым значением
            true_outputs += 1  # Увеличиваем счетчик правильных ответов

    return true_outputs, len(inputs)  # Возвращаем количество правильных ответов и общее количество данных

# Функция для тестирования модели
def test():
    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        z = weighted_z(point)  # Вычисляем z
        output = logistic_function(z)  # Получаем выходное значение
        target = targets[i]  # Получаем целевое значение
        print(f"output: {round(output, 2)}, target: {target}")  # Выводим предсказание и целевое значение

test()  # Запускаем тестирование
print("accuracy:", accuracy())  # Выводим точность модели