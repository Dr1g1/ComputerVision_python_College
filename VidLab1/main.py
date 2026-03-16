import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fft2_slike(slika):
    fft_slike = np.fft.fft2(slika)
    fft_slike = np.fft.fftshift(fft_slike)
    return fft_slike

def inverzna_fft2(log_amplituda, fazni_spektar):
    amplituda = np.expm1(log_amplituda)
    spec = fazni_spektar * amplituda
    spec = np.fft.ifftshift(spec)
    rekonstruisana_slika = np.fft.ifft2(spec)
    rekonstruisana_slika = np.real(rekonstruisana_slika)
    return rekonstruisana_slika

def fja_ublazeno(fft_slika, sum):
    for koord in sum:
        fft_slika[koord] *= 0.10  # ovako blago gasimo originalnu vrednost

def fja_direktni_susedi(fft_slika, sum_koordinate):
    for (y, x) in sum_koordinate:
        vrednost = (fft_slika[y, x - 1] +
                    fft_slika[y, x + 1] +
                    fft_slika[y - 1, x] +
                    fft_slika[y + 1, x]) / 4
        fft_slika[y, x] = vrednost

def fja_gausov_blur(fft_slika, sum):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float)
    kernel /= kernel.sum()  # normalizacija
    for (y, x) in sum:
        region = fft_slika[y - 1:y + 2, x - 1:x + 2]
        fft_slika[y - 1:y + 2, x - 1:x + 2] = region * kernel

putanja = "slika_2.png"
slika = cv2.imread(putanja)
slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

plt.title("Pocetno stanje slike")
plt.imshow(slika, cmap="gray")
plt.axis('off')
plt.show()

fft_slika = fft2_slike(slika)

log_amplituda_slika = np.log(np.abs(fft_slika)+1)
plt.title("Prikaz amplitude u logaritamskoj skali")
plt.imshow(log_amplituda_slika)
plt.axis('off')
plt.show()

sum = [(231,251), (246,261), (266,251), (281,261)]

#fja_ublazeno(fft_slika, sum)
fja_direktni_susedi(fft_slika, sum) # ovako je najcistije
#fja_gausov_blur(fft_slika, sum)

log_amplituda_nakon = np.log(np.abs(fft_slika)+1)
plt.title("Amplituda nakon uklanjanja suma - u log skali")
plt.imshow(log_amplituda_nakon)
plt.axis('off')
plt.show()

fazni_spektar = fft_slika / (np.abs(fft_slika) + 1e-12)

slika_ociscena = inverzna_fft2(log_amplituda_nakon, fazni_spektar)
plt.title("Krajnje stanje slike")
plt.imshow(slika_ociscena, cmap="gray")
plt.axis('off')
plt.show()
#slika_ociscena = Image.fromarray(np.uint8(slika_ociscena))
#slika_ociscena.save('slika_kraj.png')



