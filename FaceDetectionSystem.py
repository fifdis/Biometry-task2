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

# Функция построения графиков
def Build_Charts_func(num_e, num_c, start_pos, step, end_pos, ch_func):
    # Для финальных результатов
    stat_dct = []
    stat_dft = []
    stat_scale = []
    stat_hist = []
    stat_grad = []
    stat_dct_indv = []
    stat_dft_indv = []
    stat_scale_indv = []
    stat_hist_indv = []
    stat_grad_indv = []
    # Пороговые значения
    delta_k_h = 230
    delta_k_g = 80
    # Для тестов
    t_img_a = []
    t_hist = []
    t_grad = []
    t_dft = []
    t_dct = []
    t_scale = []
    #для эталонов
    e_img_a = []
    e_hist = []
    e_grad = []
    e_dft = []
    e_dct = []
    e_scale = []

    ind_et = 0
    ind_t = 0
    if(ch_func == 0):
        help_pos = num_e
    if(ch_func == 1):
        help_pos = 1
    if(ch_func == 2):
        help_pos = 0
    # Определение числа классов
    num_subfolders = num_c
    # Перебор подпапок (классов/людей)
    for i in range(1, num_subfolders+1, 1):
        sum_h = 0
        sum_g = 0
        sum_sim_dft = 0
        sum_sim_dct = 0
        sum_sim_scale = 0
        num_files = len([f for f in os.listdir(f"ORLdataset/s{i}") if os.path.isfile(os.path.join(f"ORLdataset/s{i}", f))])
        # Перебор эталонов
        for j in range(start_pos, end_pos + 1, step):
            res_h = 0
            res_g = 0
            fln_e = f"ORLdataset/s{i}/{j}.pgm"
            e_img = cv2.imread(fln_e, cv2.IMREAD_GRAYSCALE)
            e_img_a.append(e_img)
            e_hist.append(Hist_func(e_img))
            e_grad.append(Gradient_func(e_img))
            e_dft.append(DFT_func(e_img))
            e_dct.append(DCT_func(e_img))
            e_scale.append(Scale_func(fln_e))
            #перебор тестов
            for k in range(help_pos + 1, num_files + 1, step):
                fln_t = f"ORLdataset/s{i}/{k}.pgm"
                t_img = cv2.imread(fln_t, cv2.IMREAD_GRAYSCALE)
                t_img_a.append(t_img)
                t_hist.append(Hist_func(t_img))
                t_grad.append(Gradient_func(t_img))
                t_dft.append(DFT_func(t_img))
                t_dct.append(DCT_func(t_img))
                t_scale.append(Scale_func(fln_t))
                # Уменьшение процесса исправления и предусмотр четного/нечетного выбора
                if (ch_func == 0):
                    jjj = j - 1 + num_e * (i - 1)
                    kkk = k - num_e - 1 + (10 - num_e) * (i - 1)
                else:
                    jjj = ind_et
                    kkk = ind_t
                # Вычисление максимума на эталоне гистограммы
                in_e_h, e_m_h = max(enumerate(e_hist[jjj]), key=operator.itemgetter(1))
                # Вычисление максимума на эталоне градиента
                in_e_g, e_m_g = max(enumerate(e_grad[jjj]), key=operator.itemgetter(1))
                # Максимум для тестовой гистограммы по соотвествтующему индексу
                t_max_h = t_hist[kkk][in_e_h]
                # Максимум для тестового градиента по соотвествтующему индексу
                t_max_g = t_grad[kkk][in_e_g]
                # Разница по гистограмме
                delt_h = abs(e_m_h - t_max_h)
                # Разница по градиенту
                delt_g = abs(e_m_g - t_max_g)
                # Сравнение с границами
                if (delt_h < delta_k_h):
                    stat_hist_indv.append(1)
                    res_h += 1
                else:
                    stat_hist_indv.append(0)
                if (delt_g < delta_k_g):
                    stat_grad_indv.append(1)
                    res_g += 1
                else:
                    stat_grad_indv.append(0)
                #среднее по эталону, dft
                mean_mag_e = np.mean(e_dft[jjj])
                #среднее по тестовому, dft
                mean_mag_t = np.mean(t_dft[kkk])
                #вычисление процента совпадения
                similarity_percent_dft = mean_mag_t / mean_mag_e
                if (similarity_percent_dft > 1):
                    similarity_percent_dft = 2 - similarity_percent_dft
                sum_sim_dft += similarity_percent_dft
                #вычисление среднего по эталону, dct
                linalg_norm_e = np.linalg.norm(e_dct[jjj])
                #вычисление среднего по тестовому, dct
                linalg_norm_t = np.linalg.norm(t_dct[kkk])
                #вычисление процента совпадения
                similarity_percent_dct = linalg_norm_t / linalg_norm_e
                if (similarity_percent_dct > 1):
                    similarity_percent_dct = 2 - similarity_percent_dct
                sum_sim_dct += similarity_percent_dct
                #вычисление процента совпадения scale
                ssim = metrics.structural_similarity(e_scale[jjj], t_scale[kkk], data_range=255)
                if (ssim > 1):
                    ssim = 2 - ssim
                sum_sim_scale += ssim
                stat_dft_indv.append(similarity_percent_dft)
                stat_dct_indv.append(similarity_percent_dct)
                stat_scale_indv.append(ssim)
                ind_t += 1
            #увеличение сумм
            sum_h += res_h
            sum_g += res_g
            ind_et += 1
        #ускорение процесса исправления
        dif = num_files - num_e
        stat_hist.append(sum_h / ((dif) * num_e))
        stat_grad.append(sum_g / ((dif) * num_e))
        stat_dft.append(sum_sim_dft / ((dif) * num_e))
        stat_dct.append(sum_sim_dct / ((dif) * num_e))
        stat_scale.append(sum_sim_scale / ((dif) * num_e))
    #массивы для результатов тестовых изображений для каждой s
    stat_hist_indv_n = np.zeros((dif) * num_subfolders)
    stat_grad_indv_n = np.zeros((dif) * num_subfolders)
    stat_dft_indv_n = np.zeros((dif) * num_subfolders)
    stat_dct_indv_n = np.zeros((dif) * num_subfolders)
    stat_scale_indv_n = np.zeros((dif) * num_subfolders)
    #перебор каждого s
    for m in range(1, num_subfolders + 1, 1):
        ad = 0
        for l in range(0+(m-1)*dif, dif*m, 1):
            for o in range(0, num_e, 1):
                v = ad + dif * o + (m-1) * num_e * dif
                #расчет среднего показателя тестовых изображений относительно кол-ва эталонов
                stat_hist_indv_n[l] += stat_hist_indv[v]
                stat_grad_indv_n[l] += stat_grad_indv[v]
                stat_dft_indv_n[l] += stat_dft_indv[v]
                stat_dct_indv_n[l] += stat_dct_indv[v]
                stat_scale_indv_n[l] += stat_scale_indv[v]
            ad += 1
            stat_hist_indv_n[l] = stat_hist_indv_n[l] / num_e
            stat_grad_indv_n[l] = stat_grad_indv_n[l] / num_e
            stat_dft_indv_n[l] = stat_dft_indv_n[l] / num_e
            stat_dct_indv_n[l] = stat_dct_indv_n[l] / num_e
            stat_scale_indv_n[l] = stat_scale_indv_n[l] / num_e
    #создание фигур, внутри которых будут показываться результаты
    fig1, ((ax_1, ax_2, ax_3, ax_4, ax_5, ax_6), (ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(2, 6)
    fig2, ((axIndH, axIndG, axIndDFT, axIndDCT, axIndScale), (axH, axG, axDFT, axDCT, axScale)) = plt.subplots(2, 5)
    axIndH.set_xlabel("№ теста")
    axIndH.set_ylabel("Точность, %")
    axIndG.set_xlabel("№ теста")
    axIndDFT.set_xlabel("№ теста")
    axIndDCT.set_xlabel("№ теста")
    axIndScale.set_xlabel("№ теста")
    axH.set_xlabel("№ класса")
    axH.set_ylabel("Точность, %")
    axG.set_xlabel("№ класса")
    axDFT.set_xlabel("№ класса")
    axDCT.set_xlabel("№ класса")
    axScale.set_xlabel("№ класса")
    plt.ion()
    #для теста
    ax_1.set_title('Тест')
    i_a = ax_1.imshow(t_img_a[0], cmap='gray')
    ax_2.set_title(f'Гистограмма:{round(stat_hist_indv[0],3)}')
    h_a, = ax_2.plot(t_hist[0], color="green")
    stat_dft_indv = np.round(stat_dft_indv, decimals=3)
    ax_3.set_title(f'DFT:{round(stat_dft_indv[0],3)}')
    df_a = ax_3.imshow(t_dft[0], cmap='gray', vmin=0, vmax=255)
    stat_dct_indv = np.round(stat_dct_indv, decimals=3)
    ax_4.set_title(f'DCT:{round(stat_dct_indv[0],3)}')
    dc_a = ax_4.imshow(np.abs(t_dct[0]), vmin=0, vmax=255)
    x = np.arange(len(t_grad[0]))
    ax_5.set_title(f'Градиент:{round(stat_grad_indv[0],3)}')
    g_a, = ax_5.plot(x, t_grad[0], color="green")
    stat_scale = np.round(stat_scale, decimals=3)
    ax_6.set_title(f'Scale:{round(stat_scale[0],3)}')
    sc_a = ax_6.imshow(t_scale[0], cmap='gray')
    #для эталона
    ax1.set_title('Эталон')
    i_a_e = ax1.imshow(e_img_a[0], cmap='gray')
    ax2.set_title('Гистограмма')
    h_a_e, = ax2.plot(e_hist[0], color="green")
    ax3.set_title('DFT')
    df_a_e = ax3.imshow(e_dft[0], cmap='gray', vmin=0, vmax=255)
    ax4.set_title('DCT')
    dc_a_e = ax4.imshow(np.abs(e_dct[0]), cmap='gray', vmin=0, vmax=255)
    x_e = np.arange(len(e_grad[0]))
    ax5.set_title('Градиент')
    g_a_e, = ax5.plot(x_e, e_grad[0], color="green")
    ax6.set_title('Scale')
    sc_a_e = ax6.imshow(e_scale[0], cmap='gray')
    #для остального
    x_r_g = np.arange(len(stat_grad))
    x_r_h = np.arange(len(stat_hist))
    x_r_dft = np.arange(len(stat_dft))
    x_r_dct = np.arange(len(stat_dct))
    x_r_scale = np.arange(len(stat_scale))
    x_r_g_ind = np.arange(len(stat_grad_indv_n))
    x_r_h_ind = np.arange(len(stat_hist_indv_n))
    x_r_dft_ind = np.arange(len(stat_dft_indv_n))
    x_r_dct_ind = np.arange(len(stat_dct_indv_n))
    x_r_scale_ind = np.arange(len(stat_scale_indv_n))
    #показ фигур
    #fig1.set_size_inches(19, 5)
    fig1.show()
    #fig2.set_size_inches(19, 5)
   # fig2.show()
    #поиск числа подпапок
    num_subfolders = num_c
    #num_subfolders = len([f.path for f in os.scandir("ORLdataset") if f.is_dir()])
    #перебор подпапок

    for t in range(0, num_subfolders, 1):
        axH.plot(x_r_h[0:t+1:1], stat_hist[0:t+1:1], color="green")
        axH.set_title('Гистограмма')
        axG.plot(x_r_g[0:t+1:1], stat_grad[0:t+1:1], color="green")
        axG.set_title('Градиент')
        axDFT.plot(x_r_dft[0:t+1:1], stat_dft[0:t+1:1], color="green")
        axDFT.set_title('DFT')
        axDCT.plot(x_r_dct[0:t+1:1], stat_dct[0:t+1:1], color="green")
        axDCT.set_title('DCT')
        axScale.plot(x_r_scale[0:t+1:1], stat_scale[0:t+1:1], color="green")
        axScale.set_title('Scale')
        index = 0
        for p in range(0 + num_e * t, num_e * t + num_e, 1):
            i_a_e.set_data(e_img_a[p])
            h_a_e.set_ydata(e_hist[p])
            df_a_e.set_data(e_dft[p])
            dc_a_e.set_data(e_dct[p])
            g_a_e.set_ydata(e_grad[p])
            sc_a_e.set_data(e_scale[p])
            num_files = len([f for f in os.listdir(f"ORLdataset/s{i}") if os.path.isfile(os.path.join(f"ORLdataset/s{i}", f))])
            for m in range((0 + p * (num_files - num_e)), (num_files - num_e) * (p + 1), 1):
                i_a.set_data(t_img_a[m])
                h_a.set_ydata(t_hist[m])
                df_a.set_data(t_dft[m])
                dc_a.set_data(t_dct[m])
                g_a.set_ydata(t_grad[m])
                sc_a.set_data(t_scale[m])
                ax_2.set_title(f'Гистограмма:{round(stat_hist_indv[m],3)}')
                stat_dft_indv = np.round(stat_dft_indv, decimals=3)
                ax_3.set_title(f'DFT:{round(stat_dft_indv[m],3)}')
                stat_dct_indv = np.round(stat_dct_indv, decimals=3)
                ax_4.set_title(f'DCT:{round(stat_dct_indv[m],3)}')
                ax_5.set_title(f'Градиент:{round(stat_grad_indv[m],3)}')
                stat_scale_indv = np.round(stat_scale_indv, decimals=3)
                ax_6.set_title(f'Scale:{round(stat_scale_indv[m],3)}')
                #средний по тестам выводится, если рассмотренны все эталоны и выводится последний тест
                if((p+1 == num_e * t + num_e)):
                    put = index + t*(num_files - num_e)+1
                    axIndH.plot(x_r_h_ind[0:put:1], stat_hist_indv_n[0:put:1], color="green")
                    axIndH.set_title('Гистограмма')
                    axIndG.plot(x_r_g_ind[0:put:1], stat_grad_indv_n[0:put:1], color="green")
                    axIndG.set_title('Градиент')
                    axIndDFT.plot(x_r_dft_ind[0:put:1], stat_dft_indv_n[0:put:1], color="green")
                    axIndDFT.set_title('DFT')
                    axIndDCT.plot(x_r_dct_ind[0:put:1], stat_dct_indv_n[0:put:1], color="green")
                    axIndDCT.set_title('DCT')
                    axIndScale.plot(x_r_scale_ind[0:put:1], stat_scale_indv_n[0:put:1], color="green")
                    axIndScale.set_title('Scale')
                    index += 1
                #отрисовка фигур
                fig1.canvas.draw()
                fig1.canvas.flush_events()
               # fig2.canvas.draw()
               # fig2.canvas.flush_events()
    plt.pause(120)
    plt.close()
# Построение графиков через выбранные файлы
def Build_Charts_With_Chosen_func(filename1, filename2):
    delta_k_h = 100
    delta_k_g = 80
    res_h = 0
    res_g = 0
    sum_sim_dft = 0
    sum_sim_dct = 0
    e_img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    e_hist = Hist_func(e_img)
    e_grad = Gradient_func(e_img)
    e_dft = DFT_func(e_img)
    e_dct = DCT_func(e_img)
    e_scale = Scale_func(filename1)
    t_img = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    t_hist = Hist_func(t_img)
    t_grad = Gradient_func(t_img)
    t_dft = DFT_func(t_img)
    t_dct = DCT_func(t_img)
    t_scale = Scale_func(filename2)
    t_or_img = plt.imread(filename2, cv2.IMREAD_GRAYSCALE)
    e_or_img = plt.imread(filename1, cv2.IMREAD_GRAYSCALE)
    in_e_m, e_m_h = max(enumerate(e_hist), key=operator.itemgetter(1))
    in_e_g, e_m_g = max(enumerate(e_grad), key=operator.itemgetter(1))
    t_max_h = t_hist[in_e_m]
    t_max_g = t_grad[in_e_g]
    delt_c = abs(e_m_h - t_max_h)
    delt_g = abs(e_m_g - t_max_g)
    if (delt_c < delta_k_h):
        res_h += 1
    if (delt_g < delta_k_g):
        res_g += 1
    mean_mag_e = np.mean(e_dft)
    mean_mag_t = np.mean(t_dft)
    similarity_percent_dft = mean_mag_t / mean_mag_e
    if (similarity_percent_dft > 1):
        similarity_percent_dft = 2 - similarity_percent_dft
    sum_sim_dft += similarity_percent_dft
    linalg_norm_e = np.linalg.norm(e_dct)
    linalg_norm_t = np.linalg.norm(t_dct)
    similarity_percent_dct = linalg_norm_t / linalg_norm_e
    if (similarity_percent_dct > 1):
        similarity_percent_dct = 2 - similarity_percent_dct
    sum_sim_dct += similarity_percent_dct
    ssim = metrics.structural_similarity(e_scale, t_scale, data_range=255)
    plt.subplot(3, 6, 13)
    plt.imshow(e_or_img, cmap='gray')
    plt.title("Эталон")
    plt.subplot(3, 6, 14)
    plt.plot(e_hist, color="green")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 15)
    plt.imshow(e_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(e_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 17)
    x = np.arange(len(e_grad))
    plt.plot(x, e_grad, color="green")
    plt.title("Градиент")
    plt.subplot(3, 6, 18)
    plt.imshow(e_scale, cmap='gray')
    plt.title("Scale")
    plt.subplot(3, 6, 1)
    plt.imshow(t_or_img, cmap='gray')
    plt.title("Тест")
    plt.subplot(3, 6, 2)
    plt.plot(t_hist, color="green")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 3)
    plt.imshow(t_dft, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(t_dct), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 5)
    x = np.arange(len(t_grad))
    plt.plot(x, t_grad, color="green")
    plt.title("Градиент")
    plt.subplot(3, 6, 6)
    plt.imshow(t_scale, cmap='gray')
    plt.title("Scale")
    if (res_g != 0 and res_h != 0 and similarity_percent_dft >= 0.5 and similarity_percent_dct >= 0.5 and ssim >= 0.5):
        Show_Results_func("Найдено совпадение")
    else:
        Show_Results_func("Совпадений нет.")
    plt.show()

# Построение графиков кросс-валидации по файлу и числу эталонов
def Build_Charts_With_Chosen_Dir_func(filename, num_e):
    stat_dct = []
    stat_dft = []
    stat_scale = []
    stat_hist = []
    stat_grad = []
    stat_dct_indv = []
    stat_dft_indv = []
    stat_scale_indv = []
    stat_hist_indv = []
    stat_grad_indv = []
    delta_k_h = 230
    delta_k_g = 80
    t_img_a = []
    t_hist = []
    t_grad = []
    t_dft = []
    t_dct = []
    t_scale = []
    e_img_a = []
    e_hist = []
    e_grad = []
    e_dft = []
    e_dct = []
    e_scale = []
    for i in range(1, 2, 1):
        sum_h = 0
        sum_g = 0
        sum_sim_dft = 0
        sum_sim_dct = 0
        sum_sim_scale = 0
        cat = [f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))]
        num_files = len([f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))])
        for j in range(1, num_e + 1, 1):
            res_h = 0
            res_g = 0
            fln_e = f"{filename}/{cat[j-1]}"
            e_img = cv2.imread(fln_e, cv2.IMREAD_GRAYSCALE)
            e_img_a.append(e_img)
            e_hist.append(Hist_func(e_img))
            e_grad.append(Gradient_func(e_img))
            e_dft.append(DFT_func(e_img))
            e_dct.append(DCT_func(e_img))
            e_scale.append(Scale_func(fln_e))
            for k in range(num_e + 1, num_files + 1, 1):
                fln_t = f"{filename}/{cat[k-1]}"
                t_img = cv2.imread(fln_t, cv2.IMREAD_GRAYSCALE)
                t_img_a.append(t_img)
                t_hist.append(Hist_func(t_img))
                t_grad.append(Gradient_func(t_img))
                t_dft.append(DFT_func(t_img))
                t_dct.append(DCT_func(t_img))
                t_scale.append(Scale_func(fln_t))
                in_e_h, e_m_h = max(enumerate(e_hist[j - 1 + num_e * (i - 1)]), key=operator.itemgetter(1))
                in_e_g, e_m_g = max(enumerate(e_grad[j - 1 + num_e * (i - 1)]), key=operator.itemgetter(1))
                t_max_h = t_hist[k - num_e - 1 + (num_files - num_e) * (i - 1)][in_e_h]
                t_max_g = t_grad[k - num_e - 1 + (num_files - num_e) * (i - 1)][in_e_g]
                delt_h = abs(e_m_h - t_max_h)
                delt_g = abs(e_m_g - t_max_g)
                if (delt_h < delta_k_h):
                    stat_hist_indv.append(1)
                    res_h += 1
                else:
                    stat_hist_indv.append(0)
                if (delt_g < delta_k_g):
                    stat_grad_indv.append(1)
                    res_g += 1
                else:
                    stat_grad_indv.append(0)
                mean_mag_e = np.mean(e_dft[j - 1 + num_e * (i - 1)])
                mean_mag_t = np.mean(t_dft[k - num_e - 1 + (num_files - num_e) * (i - 1)])
                similarity_percent_dft = mean_mag_t / mean_mag_e
                if (similarity_percent_dft > 1):
                    similarity_percent_dft = 2 - similarity_percent_dft
                sum_sim_dft += similarity_percent_dft
                linalg_norm_e = np.linalg.norm(e_dct[j - 1 + num_e * (i - 1)])
                linalg_norm_t = np.linalg.norm(t_dct[k - num_e - 1 + (num_files - num_e) * (i - 1)])
                similarity_percent_dct = linalg_norm_t / linalg_norm_e
                if (similarity_percent_dct > 1):
                    similarity_percent_dct = 2 - similarity_percent_dct
                sum_sim_dct += similarity_percent_dct
                ssim = metrics.structural_similarity(e_scale[j - 1 + num_e * (i - 1)], t_scale[k - num_e - 1 + (num_files - num_e) * (i - 1)], data_range=255)
                if (ssim > 1):
                    ssim = 2 - ssim
                sum_sim_scale += ssim
                stat_dft_indv.append(similarity_percent_dft)
                stat_dct_indv.append(similarity_percent_dct)
                stat_scale_indv.append(ssim)
            sum_h += res_h
            sum_g += res_g
        stat_hist.append(sum_h/((num_files-num_e)*num_e))
        stat_grad.append(sum_g / ((num_files - num_e) * num_e))
        stat_dft.append(sum_sim_dft/((num_files - num_e) * num_e))
        stat_dct.append(sum_sim_dct/((num_files - num_e) * num_e))
        stat_scale.append(sum_sim_scale / ((num_files - num_e) * num_e))
    fig1, ((ax_1, ax_2, ax_3, ax_4, ax_5, ax_6), (ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(2, 6)
    fig2, (axH, axG, axDFT, axDCT, axScale) = plt.subplots(1, 5)
    axH.set_xlabel("№ теста")
    axH.set_ylabel("Точность, %/100")
    axG.set_xlabel("№ теста")
    axDFT.set_xlabel("№ теста")
    axDCT.set_xlabel("№ теста")
    axScale.set_xlabel("№ теста")
    plt.ion()
    ax_1.set_title('Тест')
    i_a = ax_1.imshow(t_img_a[0], cmap='gray')
    ax_2.set_title(f'Гистограмма:{round(stat_hist_indv[0],3)}')
    h_a, = ax_2.plot(t_hist[0], color="brown")
    stat_dft_indv = np.round(stat_dft_indv, decimals=3)
    ax_3.set_title(f'DFT:{round(stat_dft_indv[0],3)}')
    df_a = ax_3.imshow(t_dft[0], cmap='gray', vmin=0, vmax=255)
    stat_dct_indv = np.round(stat_dct_indv, decimals=3)
    ax_4.set_title(f'DCT:{round(stat_dct_indv[0],3)}')
    dc_a = ax_4.imshow(np.abs(t_dct[0]), vmin=0, vmax=255)
    x = np.arange(len(t_grad[0]))
    ax_5.set_title(f'Градиент:{round(stat_grad_indv[0],3)}')
    g_a, = ax_5.plot(x, t_grad[0], cmap='gray', color="brown")
    stat_scale = np.round(stat_scale, decimals=3)
    ax_6.set_title(f'Scale:{round(stat_scale[0],3)}')
    sc_a = ax_6.imshow(t_scale[0])
    ax1.set_title('Эталон')
    i_a_e = ax1.imshow(e_img_a[0], cmap='gray')
    ax2.set_title('Гистограмма')
    h_a_e, = ax2.plot(e_hist[0], color="brown")
    ax3.set_title('DFT')
    df_a_e = ax3.imshow(e_dft[0], cmap='gray', vmin=0, vmax=255)
    ax4.set_title('DCT')
    dc_a_e = ax4.imshow(np.abs(e_dct[0]), vmin=0, vmax=255)
    x_e = np.arange(len(e_grad[0]))
    ax5.set_title('Градиент')
    g_a_e, = ax5.plot(x_e, e_grad[0], cmap='gray', color="brown")
    ax6.set_title('Scale')
    sc_a_e = ax6.imshow(e_scale[0])
    x_r_g = np.arange(len(stat_grad))
    x_r_h = np.arange(len(stat_hist))
    x_r_dft = np.arange(len(stat_dft))
    x_r_dct = np.arange(len(stat_dct))
    x_r_scale = np.arange(len(stat_scale))
    axH.plot(x_r_h, stat_hist, color="brown")
    axH.set_title('Гистограмма')
    axG.plot(x_r_g, stat_grad, color="brown")
    axG.set_title('Градиент')
    axDFT.plot(x_r_dft, stat_dft, color="brown")
    axDFT.set_title('DFT')
    axDCT.plot(x_r_dct, stat_dct, color="brown")
    axDCT.set_title('DCT')
    axScale.plot(x_r_scale, stat_scale, color="brown")
    axScale.set_title('Scale')
    fig1.set_size_inches(19, 5)
    fig1.show()
    fig2.set_size_inches(19, 5)
    fig2.show()
    for t in range(0, 1, 1):
        for p in range(0 + num_e * t, num_e * t + num_e, 1):
            i_a_e.set_data(e_img_a[p])
            h_a_e.set_ydata(e_hist[p])
            df_a_e.set_data(e_dft[p])
            dc_a_e.set_data(e_dct[p])
            g_a_e.set_ydata(e_grad[p])
            sc_a_e.set_data(e_scale[p])
            num_files = len([f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))])
            for m in range((0 + p * (num_files - num_e)), (num_files - num_e) * (p + 1), 1):
                i_a.set_data(t_img_a[m])
                h_a.set_ydata(t_hist[m])
                df_a.set_data(t_dft[m])
                dc_a.set_data(t_dct[m])
                g_a.set_ydata(t_grad[m])
                sc_a.set_data(t_scale[m])
                ax_2.set_title(f'Гистограмма:{round(stat_hist_indv[m],3)}')
                stat_dft_indv = np.round(stat_dft_indv, decimals=3)
                ax_3.set_title(f'DFT:{round(stat_dft_indv[m],3)}')
                stat_dct_indv = np.round(stat_dct_indv, decimals=3)
                ax_4.set_title(f'DCT:{round(stat_dct_indv[m],3)}')
                ax_5.set_title(f'Градиент:{round(stat_grad_indv[m],3)}')
                stat_scale_indv = np.round(stat_scale_indv, decimals=3)
                ax_6.set_title(f'Scale:{round(stat_scale_indv[m],3)}')
                fig1.canvas.draw()
                fig1.canvas.flush_events()
    plt.waitforbuttonpress()
    plt.close()
# Функция получения кол-ва эталонов
def Get_eAmount_func(choosed_op):
    num_etalons = num_etalons_entry.get()
    num_classes = num_classes_entry.get()
    if (choosed_op == 0):  # случай 1. Эталоны берутся по порядку
        start_pos = 1
        step = 1
        end_pos = int(num_etalons)
    if (choosed_op == 1):  # случай 2. Эталоны нечётные
        start_pos = 1
        step = 2
        end_pos = 10
        num_etalons = str(5)
    if (choosed_op == 2):  # случай 3. Эталоны чётные
        start_pos = 2
        step = 2
        end_pos = 10
        num_etalons = str(5)
    if num_etalons.isdigit() and int(num_etalons) > 0:  # случай 4. Проверка условия ввода
        Build_Charts_func(int(num_etalons), int(num_classes), start_pos, step, end_pos, choosed_op)
    else:
        tk.showerror("Ошибка!", "Должно быть введено целое положительное число!")

# Выбор тестовых
def Select_Test_func(e_n, filename1):
    num_etalons = e_n
    if num_etalons.isdigit() and int(num_etalons) > 0:
        Build_Charts_With_Chosen_Dir_func(filename1, int(num_etalons))
    else:
        tk.showerror("WARNING", "Должно быть введено целое целое положительное число")

# Файловый диалог
def Select_Activation_func():
    filename1 = fd.askopenfilename()
    filename2 = fd.askopenfilename()
    Build_Charts_With_Chosen_func(filename1, filename2)

# Отображение результата
def Show_Results_func(text):
    msg = text
    mb.showinfo("Результат", msg)

# Кросс-валидация
def Select_CrossValidation():
    root1 = tk.Tk()
    root1.geometry('300x150')
    plot_button2 = tk.Button(root1, text="Нечетные эталоны",
                             command=lambda: Get_eAmount_func(int(1)))
    plot_button2.pack()
    plot_button3 = tk.Button(root1, text="Четные эталоны", command=lambda: Get_eAmount_func(int(2)))
    plot_button3.pack()

# Смена цвета для кнопок при наведении
def on_enter(e):
    e.widget['background'] = 'blue'
def on_leave(e):
    e.widget['background'] = 'SystemButtonFace'

# Главное окно
root = tk.Tk()
root.geometry("640x400")
root.title("Face Recognition System")
# Задание параметров
num_etalons_label = tk.Label(root, text="Введите кол-во эталонов:", font='Times 12')
num_etalons_label.pack()
num_etalons_entry = tk.Entry(root)
num_etalons_entry.pack()

num_classes_label = tk.Label(root, text="Введите кол-во классов сравнения:", font='Times 12')
num_classes_label.pack()
num_classes_entry = tk.Entry(root)
num_classes_entry.pack()


# Выбор из доступных действий
plot_button1 = tk.Button(root, text="Обучение", width=40, height=2, pady=10, command=lambda :Get_eAmount_func(int(0)))
plot_button1.pack()
plot_button1.bind("<Enter>", on_enter)
plot_button1.bind("<Leave>", on_leave)
plot_button2 = tk.Button(root, text="Выбрать изображения", width=40, height=2, pady=10, command=Select_Activation_func)
plot_button2.pack()
plot_button2.bind("<Enter>", on_enter)
plot_button2.bind("<Leave>", on_leave)
plot_button4 = tk.Button(root, text="Кросс-валидация", width=40, height=2, pady=10, command=Select_CrossValidation)
plot_button4.pack()
plot_button4.bind("<Enter>", on_enter)
plot_button4.bind("<Leave>", on_leave)
root.mainloop()

