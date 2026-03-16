import cv2
import numpy as np
import matplotlib.pyplot as plt

def morfoloska_rekonstrukcija(marker, maska):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #kernel koji ce da se koristi za dilataciju
    pret_marker = np.zeros_like(marker)
    tren_marker = marker.copy()
    while not np.array_equal(tren_marker, pret_marker):
        pret_marker = tren_marker.copy()
        tren_marker = cv2.dilate(tren_marker, kernel3)
        tren_marker = cv2.min(tren_marker, maska)

    return tren_marker


# 1) ucitavanje slike:
slika = cv2.imread("coins.png")
slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(slika_rgb)
plt.title("Pocetna slika")
plt.show()

# 2) segmentacija svih novcica:
slika_gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(slika_gray, cmap = "gray")
plt.title("Slika nakon BGR2GRAY")
plt.show()

# pravimo histogram da bismo mogli rucno da nadjemo threshold
hist, bins = np.histogram(slika_gray, bins=256, range=(0,256))
plt.bar(bins[:-1], hist)
plt.xlabel("Intenzitet")
plt.ylabel("Broj piksela")
plt.show()

_, coins_maska = cv2.threshold(slika_gray, 209, 255, cv2.THRESH_BINARY_INV)
# cv2.threshold(src, thresh, maxval, type)
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(coins_maska, cmap = "gray")
plt.title("Rezultat fje threshold")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # sto je kernel vecih dimenzija operacija postaje jaca

# OPCIJA 1: da bi se maska popunila koristi se morfoloska operacija "closing" ili "dilation + fill holes"
coins_maska2 = cv2.morphologyEx(coins_maska, cv2.MORPH_CLOSE, kernel)

#OPCIJA 2:
# dilatacija
#dilated = cv2.dilate(coins_maska, kernel, iterations=1)
# erozija
#coins_maska2 = cv2.erode(dilated, kernel, iterations=1)

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(coins_maska2, cmap = "gray")
plt.title("Rezultat maske nakon popunjenih rupa")
plt.show()

# PRELAZAK U HSV PROSTOR BOJA:
hsv_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_slika)

# mozemo vizuelno da analiziramo histogram S kanala:
plt.hist(s.ravel(), bins=256, color='orange')
plt.title("Histogram zasićenosti (S kanal)")
plt.xlabel("Vrednost zasićenosti")
plt.ylabel("Broj piksela")
plt.show()
_, bakarni_marker = cv2.threshold(s,80, 255, cv2.THRESH_BINARY)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
bakarni_marker2 = cv2.morphologyEx(bakarni_marker, cv2.MORPH_OPEN, kernel2)
# ovaj marker je binarna slika koja pokazuje gde se nalazi bakarni novcic na slici
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(bakarni_marker2, cmap = "gray")
plt.title("Bakarni marker")
plt.show()

bakarni_novcic_maska = morfoloska_rekonstrukcija(bakarni_marker2, coins_maska2)
konacna_maska = np.zeros_like(bakarni_novcic_maska)
konacna_maska[bakarni_novcic_maska > 0] = 255

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(konacna_maska, cmap = "gray")
plt.title("Konacna slika - rezultat")
plt.show()

slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
bakarni_only = cv2.bitwise_and(slika_rgb, slika_rgb, mask=konacna_maska)

plt.figure(figsize=(10,5))
plt.imshow(bakarni_only)
plt.title("Samo bakarni novčić")
plt.axis("off")
plt.show()
