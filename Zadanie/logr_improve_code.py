from random import randint  # Импортируем функцию randint для генерации случайных целых чисел
from math import e, log  # Импортируем константу e и функцию логарифма

def logistic_function(z):  # Определяем логистическую функцию
    return 1 / (1 + e ** (-z))  # Возвращаем значение логистической функции

def logistic_error(outputs, targets):  # Определяем функцию для вычисления ошибки логистической регрессии
    error = 0  # Инициализируем переменную для хранения ошибки

    for i, point in enumerate(inputs):  # Проходим по всем входным данным
        if outputs[i] == 1:  # Если выход равен 1
            outputs[i] = 0.99999  # Устанавливаем его в 0.99999 для избежания логарифма нуля

        if outputs[i] == 0:  # Если выход равен 0
            outputs[i] = 0.00001  # Устанавливаем его в 0.00001 для избежания логарифма нуля
        
        # Вычисляем ошибку с использованием логарифмической функции
        error -= targets[i] * log(outputs[i], e) - (1 - targets[i]) * log(1 - outputs[i], e)

    return error / len(targets)  # Возвращаем среднюю ошибку


class LogisticRegression:  # Определяем класс для логистической регрессии
    def init(self, features_num):  # Конструктор класса
        # +1 для смещения, смещение является последним весом
        self.weights = [randint(-100, 100) / 100 for _ in range(features_num + 1)]  # Инициализируем веса случайными значениями


    def forward(self, input_features):  # Метод для прямого прохода
        output = 0  # Инициализируем выходное значение

        for i, feature in enumerate(input_features):  # Проходим по всем входным признакам
            output += self.weights[i] * feature  # Вычисляем взвешенную сумму

        return logistic_function(output + self.weights[-1])  # Возвращаем результат логистической функции с учетом смещения


    def train(self, inp, output, target, samples_num, lr):  # Метод для обучения модели
        for j in range(len(self.weights) - 1):  # Обновляем веса для всех признаков
            self.weights[j] += lr * (1 / samples_num) * (target - output) * inp[j]  # Обновляем вес

        self.weights[-1] += lr * (1 / samples_num) * (target - output)  # Обновляем вес смещения


    def forward_list(self, inputs):  # Метод для обработки списка входных данных
        outputs = []  # Инициализируем список выходов

        for inp in inputs:  # Проходим по всем входным данным
            output = self.forward(inp)  # Получаем выход для каждого входа
            outputs.append(output)  # Добавляем выход в список

        return outputs  # Возвращаем список выходов


    def fit(self, inputs, targets, epochs=100, lr=0.1):  # Метод для обучения модели
        for epoch in range(epochs):  # Проходим по всем эпохам
            outputs = []  # Инициализируем список выходов

            for i, inp in enumerate(inputs):  # Проходим по всем входным данным
                output = self.forward(inp)  # Получаем выход
                outputs.append(output)  # Добавляем выход в список

                self.train(inp, output, targets[i], len(inputs), lr)  # Обучаем модель

            print(f"epoch: {epoch}, error: {logistic_error(outputs, targets)}")  # Выводим информацию об эпохе и ошибке



def accuracy(outputs, targets):  # Функция для вычисления точности
    true_outputs = 0  # Инициализируем счетчик правильных ответов

    for i, output in enumerate(outputs):  # Проходим по всем выходам
        if round(output) == targets[i]:  # Если округленный выход совпадает с целевым значением
            true_outputs += 1  # Увеличиваем счетчик правильных ответов

    return true_outputs, len(inputs)  # Возвращаем количество правильных ответов и общее количество входов
if __name__ == '__main__':  # Проверяем, является ли этот файл основным
    x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]  # Первые наборы данных
    y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]  # Соответствующие целевые значения
    x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]  # Вторые наборы данных
    y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]  # Соответствующие целевые значения

    inputs = [(x1[i], y1[i]) for i in range(len(x1))]  # Объединяем входные данные в кортежи
    targets = [0 for i in range(len(x1))]  # Создаем список целевых значений для первого набора данных
    inputs += [(x2[i], y2[i]) for i in range(len(x2))]  # Добавляем второй набор данных
    targets += [1 for i in range(len(x2))]  # Создаем список целевых значений для второго набора данных

    logr_model = LogisticRegression(features_num=2)  # Создаем экземпляр модели логистической регрессии
    logr_model.fit(inputs, targets, epochs=100, lr=0.1)  # Обучаем модель

    outputs = logr_model.forward_list(inputs)  # Получаем выходы для всех входных данных

    print(logr_model.weights)  # Выводим веса модели
    print("accuracy:", accuracy(outputs, targets))  # Выводим точность модели