from random import randint  # Импортируем функцию randint из модуля random для генерации случайных целых чисел
# Импортируем pyplot из matplotlib для построения графиков
from matplotlib import pyplot as plt # type: ignore

# Определяем функцию data_points с параметрами: количество образцов, координаты класса и уровень шума
def data_points(n_samples, class_point, noise=1):  
# Вложенная функция для генерации смещенной точки
    def offset_point(): 
        offset_x = randint(-100 * noise, noise * 100) / 100  # Генерируем случайное смещение по оси X
        offset_y = randint(-100 * noise, noise * 100) / 100  # Генерируем случайное смещение по оси Y
        x = class_point[0] + offset_x  # Вычисляем новое значение X с учетом смещения
        y = class_point[1] + offset_y  # Вычисляем новое значение Y с учетом смещения
        return x, y  # Возвращаем координаты смещенной точки

# Генерируем список смещенных точек
    points = [offset_point() for _ in range(n_samples)]
    # Извлекаем координаты X из списка точек
    x_list = [points[i][0] for i in range(n_samples)]
    # Извлекаем координаты Y из списка точек
    y_list = [points[i][1] for i in range(n_samples)]

# Возвращаем списки координат X и Y
    return x_list, y_list

# Генерируем 10 точек для первого класса с центром в (1, 1) и уровнем шума 0.9
x1_list, y1_list = data_points(n_samples=10, class_point=(1, 1), noise=0.9)
# Генерируем 7 точек для второго класса с центром в (4, 2.5) и уровнем шума 2.5
x2_list, y2_list = data_points(n_samples=7, class_point=(4, 2.5), noise=2.5)

# Выводим координаты точек первого класса
print(x1_list, y1_list)
# Выводим координаты точек второго класса
print(x2_list, y2_list)  

# Строим график для точек первого класса красным цветом
plt.scatter(x=x1_list, y=y1_list, color='red')
# Строим график для точек второго класса зеленым цветом
plt.scatter(x=x2_list, y=y2_list, color='green')

# Отображаем график
plt.show()