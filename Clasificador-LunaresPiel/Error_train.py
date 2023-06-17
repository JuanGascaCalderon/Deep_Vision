# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:04:23 2023

@author: JUAN PABLO GASCA
"""
#Import library's
import cv2  # Librería para procesamiento de imágenes
import numpy as np  # Librería para cálculos numéricos
import glob  # Librería para trabajar con rutas de archivos
import matplotlib.pyplot as plt  # Librería para visualización de datos
from matplotlib import cm  # Módulo para mapas de colores en matplotlib
#from mpl_toolkits.mplot3d import Axes3D  # Módulo para gráficos 3D en matplotlib

#Show the picture
#img1 = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\dysplasticNevi\train\dysplasticNevi1.jpg")

def getFeatures(img):
    #Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir imagen a escala de grises
    #Otso Method
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # Calcular umbral utilizando el método de Otsu
    mask = np.uint8(1 * (gray < threshold))  # Crear una máscara binaria basada en el umbral
    
    # Calcular las características B, G y R normalizadas utilizando la máscara
    B = (1 / 255) * np.sum(img[:, :, 0] * mask) / np.sum(mask)
    G = (1 / 255) * np.sum(img[:, :, 1] * mask) / np.sum(mask)
    R = (1 / 255) * np.sum(img[:, :, 2] * mask) / np.sum(mask)
    
    return [B, G, R]  # Devolver las características como una lista

#Generation of features dataset
paths = [r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\dysplasticNevi\train",
         r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\spitzNevus\train"]

labels = []  # Etiquetas
features = []  # Características

# Recorrer las rutas de las carpetas de datos
for label, path in enumerate(paths):
    for filename in glob.glob(path + "\\*.jpg"):  # Obtener la lista de archivos .jpg en la carpeta
        img = cv2.imread(filename)  # Leer la imagen
        features.append(getFeatures(img))  # Obtener características de la imagen
        labels.append(label)  # Asignar la etiqueta a la imagen

# Convertir las listas de características y etiquetas en arrays de NumPy
features = np.array(features)
labels = np.array(labels)

labels = 2 * labels - 1  # Transformar las etiquetas a -1 y 1

# Data Visualization on the space features
fig = plt.figure()  # Crear una figura
ax = fig.add_subplot(111, projection='3d')  # Agregar un subplot 3D a la figura

# Graficar los puntos de características en función de las etiquetas
for i, features_row in enumerate(features):
    if labels[i] == -1:
        ax.scatter(features_row[0], features_row[1], features_row[2], marker='*', c='k')
    else:
        ax.scatter(features_row[0], features_row[1], features_row[2], marker='*', c='r')

ax.set_xlabel('B')  # Etiqueta del eje x
ax.set_ylabel('G')  # Etiqueta del eje y
ax.set_zlabel('R')  # Etiqueta del eje z

# Functions errors about the hiperplane constants
subFeatures = features[:, 1::]  # Extraer subconjunto de características sin la primera columna
loss = []  # Lista para almacenar los errores

# Loss Function
for w1 in np.linspace(-6, 6, 100):
    for w2 in np.linspace(-6, 6, 100):
        totalError = 0
        for i, features_row in enumerate(subFeatures):
            sample_error = (w1 * features_row[0] + w2 * features_row[1] - labels[i]) ** 2
            totalError += sample_error
        loss.append([w1, w2, totalError])

loss = np.array(loss)  # Convertir la lista de errores en un array de NumPy

# See the loss in a plot
fig = plt.figure()  # Crear una nueva figura
ax1 = fig.add_subplot(111, projection='3d')  # Agregar un subplot 3D a la figura

# Graficar el error en forma de superficie
ax1.plot_trisurf(loss[:, 0], loss[:, 1], loss[:, 2], cmap=cm.jet, linewidth=0)
ax1.set_xlabel('w1')  # Etiqueta del eje x
ax1.set_ylabel('w2')  # Etiqueta del eje y
ax1.set_zlabel('loss')  # Etiqueta del eje z

# Hiperplano que separa las dos clases de forma óptima
A = np.zeros((4, 4))  # Matriz de constantes
b = np.zeros((4, 1))  # Vector de constantes

for i, features_row in enumerate(features):
    x = np.append([1], features_row)  # Agregar un 1 al vector de características
    x = x.reshape((4, 1))  # Redimensionar el vector de características
    y = labels[i]  # Obtener la etiqueta
    
    A = A + x * x.T  # Actualizar la matriz de constantes
    b = b + x * y  # Actualizar el vector de constantes

inverseA = np.linalg.inv(A)  # Calcular la inversa de la matriz A

W = np.dot(inverseA, b)  # Calcular los coeficientes del hiperplano
X = np.arange(0, 1, 0.1)  # Rango de valores para el eje x
Y = np.arange(0, 1, 0.1)  # Rango de valores para el eje y

X, Y = np.meshgrid(X, Y)  # Crear una malla para los valores de x y y

Z = -(W[1] * X + W[2] * Y + W[0]) / W[3]  # Calcular los valores de z

ax.plot_surface(X, Y, Z, cmap=cm.Blues)  # Graficar el hiperplano
plt.show()  # Mostrar el gráfico



# Error de entrenamiento
prediction = 1 * (W[0] + np.dot(features, W[1::])) >= 0
# Realiza la predicción del modelo: multiplica las características por los pesos y suma el sesgo.
# Luego, compara si la suma es mayor o igual a cero y convierte los valores True en 1 y los valores False en 0.

prediction = 2 * prediction - 1
# Convierte las predicciones binarias en predicciones numéricas, donde True se mapea a 1 y False se mapea a -1.
# Multiplica por 2 y resta 1 para realizar este mapeo.

error = np.sum(prediction != labels.reshape(-1, 1)) / len(labels)
# Calcula el error de entrenamiento comparando las predicciones con las etiquetas reales.
# Crea una matriz de comparación donde se verifica si las predicciones son diferentes de las etiquetas.
# Suma los valores True en la matriz de comparación y divide por la longitud de las etiquetas para obtener la proporción de errores.

accuracy = 1 - error # Obtiene la proporción de predicciones correctas.


#Predicción para una imagen
path_img = r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\spitzNevus\train\spitzNevus2.jpg"
img = cv2.imread(path_img)

feature_vector = np.array(getFeatures(img))
# Llama a la función 'getFeatures' para extraer las características de la imagen y las almacena en un vector.
# Luego convierte el vector en un array NumPy y lo guarda en la variable 'feature_vector'

result = np.sign(W[0] + np.dot(feature_vector, W[1::]))
# Realiza una clasificación utilizando un vector de características y una matriz de pesos.
# Calcula el producto escalar entre 'feature_vector' y 'W[1::]' (todos los elementos de 'W' excepto el primer elemento),
# le suma 'W[0]' y aplica la función 'sign' para obtener el signo del resultado.
# El resultado se guarda en la variable 'result'

if result == -1:
    print('Es un dysplasticNevi')
else:
    print('Es un spitzNevus')
# Comprueba el valor de 'result'. Si es igual a -1, imprime "Es un dysplasticNevi".
# De lo contrario, imprime "Es un spitzNevus".
# Estos mensajes indican el resultado de la clasificación basado en el valor de 'result'.

    
    
    
    
    