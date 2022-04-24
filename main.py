import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot():
    plt.gcf().canvas.set_window_title("График")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    for i in range(len(data[0])):
        plt.hlines(data[0][i], data[1][i], data[2][i])
    legend_obj = plt.legend()
    legend_obj.set_draggable(True)
    plt.show()


def plot1():
    plt.gcf().canvas.set_window_title("График")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    for i in range(len(xp)-1):
        ax.plot([xp[i],xp[i+1]], [yp[i],yp[i+1]], marker='o')
    legend_obj = plt.legend()
    legend_obj.set_draggable(True)
    plt.show(block=False)


def plot2():
    plt.gcf().canvas.set_window_title("График")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    for i in range(len(xp2)-1):
        ax.bar(np.arange(xp2[i], xp2[i+1]), yp[i], align = 'edge', width=abs(xp2[i+1]-xp2[i]))
    legend_obj = plt.legend()
    legend_obj.set_draggable(True)
    plt.show(block=False)


read = open("input.txt", "rt", encoding='UTF-8')
inpt = []
for i in range(20):
    x = read.readline()
    inpt.append(float(x))
inpt.sort()
print("Вариационный ряд:")
print(* inpt)
print("Экстремальные значения")
print("MIN = ", min(inpt))
print("MAX = ", max(inpt))
print("Размах выбоки")
print(round(max(inpt) - min(inpt),3))
mat = 0
for i in range(len(inpt)):
    mat += inpt[i]/len(inpt)
print("Оценка мать. ожидания:")
print(round(mat,5))
dis = 0
for i in range(len(inpt)):
    dis += (inpt[i]**2)/len(inpt)
dis = dis - mat**2
print("Дисперсия:")
print(round(dis,5))
print("Среднеквадратичное отклонение")
print(round(dis**0.5,5))
x = [[], []]
x[0].append(inpt[0])
x[1].append(1)
for i in range(1, len(inpt)):
    if(inpt[i] == inpt[i-1]):
        x[1][-1] += 1
        continue
    x[0].append(inpt[i])
    x[1].append(1)
p = 1/len(inpt)
print("Выборка:")
for i in range(len(x[0])):
    print(x[0][i], ' ', x[1][i], ' ', round(x[1][i]*p,2))
print("Функция:")
data = [[], [], []]
labels = []
print("x <=", ' ', x[0][0], ' ', '->0,00')
labels.append(str("x<=" + str(x[0][0]) + '-> 1'))
data[0].append(0)
data[1].append(x[0][0]-0.5)
data[2].append(x[0][0])
sumpi = 0
for i in range(1, len(x[0])):
    sumpi += x[1][i-1] * p
    sumpi = round(sumpi, 2)
    labels.append(str(str(x[0][i-1]) + "<x<="  + str(x[0][i]) + '->' + str(sumpi)))
    data[0].append(sumpi)
    data[1].append(x[0][i - 1])
    data[2].append(x[0][i])
    print(x[0][i-1], "< x <=", ' ', x[0][i], ' ', '-> ', sumpi)
print("x <=", ' ', x[0][-1], ' ', '-> 1')
labels.append(str("x<=" + str(x[0][-1]) + '->1'))
data[0].append(1)
data[1].append(x[0][-1])
data[2].append(x[0][-1]+0.5)
plot()
h = (inpt[-1] - inpt[0])/(1+(np.log(len(inpt))/np.log(2)))
start = inpt[0] + h/2
m = int((1+(np.log(len(inpt))/np.log(2))))
xp = []
yp = []
xp2 = []
for i in range(m):
    count = 0
    for value in inpt:
        if(value >= start and value <(start + h)):
            count += 1
    print(round(start,2), ' : ', round(start+h,2), ' -> ', count/len(inpt))
    xp2.append(start)
    xp.append(round(start+h/2,2))
    yp.append(count/len(inpt))
    start += h
xp2.append(start)
print(xp2, yp)
plot1()
plot2()