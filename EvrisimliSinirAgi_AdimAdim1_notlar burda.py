# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:41:45 2022

@author: kbebim05
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)

# random.seed komutuyla asagıya yapacagımız islemleri tutmaya calıstık
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0), (pad, pad), (pad,pad), (0,0)), "constant", constant_values=0)
    
    return X_pad
"""
random olarak 4x3x3x2 lik x olursutrduk. Bunada x,2 lik
bir zero peding işlemi yaptık.

x_pad= peding işlemi uygulanmıs olan matris sonucta
x_pad.shape = (4, 7, 7, 2) cıkıyor.

"""
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)

print("x.shape =", x.shape)
print("x_pad.shape =", x_pad.shape)
print("x[1,1] =", x[1,1])
print("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1,2)
axarr[0].set_title("x")
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title("x_pad")
axarr[1].imshow(x_pad[0,:,:,0])

# Simdi 1 adet convolusyon islemi uygulayalım

"""a_slilce_prev= giriş matrisi
w = agırlık matrisi ve bias=b değeri
z = cıkıs matrisi
"""
def conv_single_step(a_slice_prev, W,b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = float(b)+Z
    return Z
"""
değerlerin tutulmasını istiyorum
-a matrisinin boyutu
-W(filtre-agırlık) matrisinin boyutu-a ile W nin kanal
sayıları eşit olmalı(3 burada)
-Z= Giris matirsinin filtrelerle carpılmıs bias değeriyle
toplanmıs hali
"""
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)

# İleri Yönlü Hesaplama

"""
n_H yükselik, n_W genişlik n_C kanal sayısı
hiperparametreler stride ve pading degeridir
"""
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
"""
Her zaman uyguladıgımız evrişimin çıkıs boyutu formülü
(((n+2p-f)/s)+1)
Çıkıs matrisi Z i sıfır baslangıc kosulları ile olusturalım
A_prev matirsine for döngüsü ile pixel ekleme işlemi
yapıcaz.

"""
n_H = int(((n_H_prev-f+2*pad)/stride)+1)
n_W = int(((n_W_prev-f+2*pad)/stride)+1)

Z = np.zeros([m, n_H, n_W, n_C])

A_prev_pad = zero_pad(a_prev, pad)

for i in range(m):
    a_prev_pad = A_prev_pad[i]
    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                
                vert_start = h*stride
                vert_end = vert_start +f
                horiz_start = w*stride
                horiz_end = horiz_start +f
            
                s_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end]
"""evrişim islemini gerceklestirecegimiz kısım"""
                Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])

"""çıkıs değerinin boyutunun doğrulugunu kontrol edelim"""
assert(Z.shape ==(m, n_H, n_W, n_C))
"""geriye yayılım işlemini uygulayabilmemiz için bu değerleri saklamamız gerekiyor"""

cache = (A_prev, W, b, hparameters)
return Z, cache
"""ekranda işlemleri görelim"""

np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)

hparameters = {"pad" :2,
               "stride":2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

print("Znin ortalama =", np.mean(Z))
print("Z[3, 2, 1]=",Z[3, 2, 1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


