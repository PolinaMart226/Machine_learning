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
# Создаем список входных данных для класса 0
inputs = [(X1[0][i], X1[1][i]) for i in range(len(X1[0]))]
# Создаем список целевых значений для класса 0
targets = [0 for _ in range(len(X1[0]))] 
# Добавляем входные данные для класса 1
inputs += [(X2[0][i], X2[1][i]) for i in range(len(X2[0]))] 
 # Добавляем целевые значения для класса 1
targets += [1 for _ in range(len(X2[0]))]

# Инициализируем веса случайными значениями
weights = [randint(-100, 100) / 100 for _ in range(3)]  

# Функция для вычисления взвешенной суммы
def weighted_z(point):  
  # Умножаем каждую координату на соответствующий вес
    z = [item * weights[i] for i, item in enumerate(point)]
  # Возвращаем сумму и добавляем смещение (bias)
    return sum(z) + weights[-1]

 # Логистическая функция
def logistic_function(z):
  # Возвращаем значение логистической функции
    return 1 / (1 + e ** (-z))  

# Функция для вычисления ошибки логистической регрессии
def logistic_error():
 # Список для хранения ошибок
    errors = []
 # Проходим по всем входным данным
    for i, point in enumerate(inputs): 
     # Вычисляем взвешенную сумму
        z = weighted_z(point)
      # Получаем выходное значение
        output = logistic_function(z)
       # Получаем целевое значение
        target = targets[i]  

     # Если выход равен 1, заменяем его на 0.99999 для избежания логарифма нуля
        if output == 1:  
            output = 0.99999
         # Если выход равен 0, заменяем его на 0.00001
        if output == 0:  
            output = 0.00001

     # Вычисляем ошибку
        error = -(target * log(output, e) - (1 - target) * log(1 - output, e)) 
     # Добавляем ошибку в список
        errors.append(error) 

 # Возвращаем среднюю ошибку
    return sum(errors) / len(errors)

# Гиперпараметры
# Ввели значение скорости обучения
lr = 0.1 
# Ввели значение количество эпох
num_epochs = 100 

# Обучение модели
# Проходим по всем эпохам
for epoch in range(num_epochs):  
  # Проходим по всем входным данным
    for i, point in enumerate(inputs):
     # Вычисляем взвешенную сумму
        z = weighted_z(point)
     # Получаем выходное значение
        output = logistic_function(z)
     # Получаем целевое значение
        target = targets[i]

      # Обновляем веса для входных данных
        for j in range(len(weights) - 1):
         # Обновляем вес
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))

      # Обновляем смещение (bias)
        weights[-1] -= lr * (output - target) * (1 / len(inputs))

    # Вывод значений epoch, error, x1, x2, bias, output и target
  # Вычисляем ошибку
    error = logistic_error()
    print(f"epoch: {epoch}, error: {error}, x1: {inputs[i][0]}, x2: {inputs[i][1]}, bias: {weights[-1]}, output: {round(output, 2)}, target: {target}")  # Выводим информацию об эпохе
  # Выводим финальные веса
print(weights) 

# Функция для вычисления точности модели
def accuracy():
  # Счетчик правильных ответов
    true_outputs = 0 
 # Проходим по всем входным данным
    for i, point in enumerate(inputs):
     # Вычисляем взвешенную сумму
        z = weighted_z(point)
     # Получаем выходное значение
        output = logistic_function(z)
     # Получаем целевое значение
        target = targets[i]

      # Если округленное выходное значение совпадает с целевым
        if round(output) == target:
          # Увеличиваем счетчик правильных ответов
            true_outputs += 1 

 # Возвращаем количество правильных ответов и общее количество данных
    return true_outputs, len(inputs)

# Функция для тестирования модели
def test(): 
 # Проходим по всем входным данным
    for i, point in enumerate(inputs):
     # Вычисляем взвешенную сумму
        z = weighted_z(point)
      # Получаем выходное значение
        output = logistic_function(z)
     # Получаем целевое значение
        target = targets[i]
     # Выводим выходное значение и целевое значение
        print(f"output: {round(output, 2)}, target: {target}") 
     
# Запускаем тестирование
test() 
 # Выводим точность модели
print("accuracy:", accuracy())

# Графическая интерпретация
# Устанавливаем размер графика
plt.figure(figsize=(10, 6))
 # Отображаем точки первого класса
plt.scatter(X1[0], X1[1], color='purple', label='Class 0 (X1)') 
 # Отображаем точки второго класса
plt.scatter(X2[0], X2[1], color='violet', label='Class 1 (X2)')
# Заголовок графика
plt.title('Logistic regression')
# Подпись оси X
plt.xlabel('X values')
# Подпись оси Y
plt.ylabel('Y values')
# Отображаем легенду
plt.legend()
# Включаем сетку
plt.grid() 
# Показываем график
plt.show()
