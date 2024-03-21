# Импорт библиотек
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.filedialog as fd
import operator
import os
from skimage import io, measure, transform, metrics
from skimage.measure import block_reduce
from skimage.color import rgb2gray
from os.path import dirname as up

# Градиент
def Gradient_func (file):
    # Размер окна и величины смещения
    ksize = 3
    dx, dy = 1, 1
    # Вычисление градиента
    gradient_x = cv2.Sobel(file, cv2.CV_32F, dx, 0, ksize=ksize)
    gradient_y = cv2.Sobel(file, cv2.CV_32F, 0, dy, ksize=ksize)
    # Вычисление абсолютного значения градиента
    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)
    # Вычисление итогового градиента
    gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)
    SumGrad = []
    for i in range(0, len(gradient), 1):
        SumGrad.append(round(sum(gradient[i]) / len(gradient[i]), 1))
    return SumGrad

# DFT
def DFT_func(file):
    # Врименение двумерного дискретного преобразования Фурье (DFT)
    dft = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Сдвиг нулевых частот в центр (циклическая свертка)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

# DCT
def DCT_func(file):
    # Применение двумерного дискретного косинусного преобразования (DCT)
    dct = cv2.dct(np.float32(file))
    return dct

# Гистограмма
def Hist_func(file):
    # Вычисление гистограммы
    histg = cv2.calcHist([file], [0], None, [256], [0, 256])
    return histg

# Scale
def Scale_func(file):
    # Поиск картинки по пути файла
    img = io.imread(file, as_gray=True)
    # Измененеие размера картинки
    img_res = transform.resize(img, (40, 40))
    return img_res
