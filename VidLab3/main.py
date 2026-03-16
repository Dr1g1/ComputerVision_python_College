import cv2 as cv
import numpy as np


# David Lowe je empirijski dokazao da je 0.7 najbolja vrednost jer nema previse matcheva i dovoljno je jasna razlika
def detekcija_i_matchovanje(slika1, slika2, ratio_thresh=0.7, scale=1.0):
    if scale != 1.0:
        skalirana_slika1 = cv.resize(slika1, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        skalirana_slika2 = cv.resize(slika2, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    else:
        skalirana_slika1, skalirana_slika2 = slika1, slika2

    # sift detekcija i isracunavanje deskriptora
    sift = cv.SIFT_create()
    ktacke1, deskriptori1 = sift.detectAndCompute(skalirana_slika1, None)
    ktacke2, deskriptori2 = sift.detectAndCompute(skalirana_slika2, None)

    if deskriptori1 is None or deskriptori2 is None:
        return None, None

    FLANN_INDEX_KDTREE = 1
    indeks_parametri = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    parametri_trazenja = dict(checks=50)
    flann = cv.FlannBasedMatcher(indeks_parametri, parametri_trazenja)

    mecovi = flann.knnMatch(deskriptori1, deskriptori2, k=2)
    tacke1 = []
    tacke2 = []

    for m_n in mecovi:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            (x1, y1) = ktacke1[m.queryIdx].pt
            (x2, y2) = ktacke2[m.trainIdx].pt
            if scale != 1.0: # vracamo originalni scale
                tacke1.append((x1 / scale, y1 / scale))
                tacke2.append((x2 / scale, y2 / scale))
            else:
                tacke1.append((x1, y1))
                tacke2.append((x2, y2))

    if len(tacke1) == 0:
        return None, None

    return np.float32(tacke1), np.float32(tacke2)

def izracunaj_homografiju(tacke_src, tacke_dst, ransac_thresh=5.0):
    if tacke_src is None or tacke_dst is None:
        return None, None
    if len(tacke_src) < 4 or len(tacke_dst) < 4:
        return None, None
    H, maska = cv.findHomography(tacke_src, tacke_dst, cv.RANSAC, ransac_thresh)
    return H, maska

# Canvas / homographies assembly (mapira svaku sliku prema koordinatama srednje slike)
def priprema_kanvasa_i_transformacija(slike, homografije):
    svi_uglovi = []
    for slika, H in zip(slike, homografije):
        h, w = slika.shape[:2]
        uglovi = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32).reshape(-1, 1, 2)
        if H is None:
            tacke = uglovi
        else:
            tacke = cv.perspectiveTransform(uglovi, H)
        svi_uglovi.append(tacke)

    svi_uglovi = np.concatenate(svi_uglovi, axis=0)
    x_min, y_min = np.int32(svi_uglovi.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(svi_uglovi.max(axis=0).ravel() + 0.5)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0 # isracunavamo pomeraj

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    H_canvas = [(T @ H).astype(np.float64) for H in homografije]

    return H_canvas, (canvas_h, canvas_w), (tx, ty)

# preslikava slike na kanvas
def preslikaj_avg_blend(slike, homografije, velicina_kanvasa):
    canvas_h, canvas_w = velicina_kanvasa
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for slika, H in zip(slike, homografije):
        warped = cv.warpPerspective(slika, H, (canvas_w, canvas_h))
        maska = (warped.sum(axis=2) > 0).astype(np.float32)
        accum += warped.astype(np.float32)
        weight += maska

    weight3 = weight[:, :, None]
    weight3[weight3 == 0] = 1.0
    rezultat = (accum / weight3).astype(np.uint8) # racunanje proseka
    return rezultat

# Feather blending - postepeno se gasi uticaj slike ka njenim ivicama
def preslikaj_feather_blend(slike, homografije, vel_kanvas):
    canvas_h, canvas_w = vel_kanvas
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for slika, H in zip(slike, homografije): # svaka slika se posebno preslikava
        warped = cv.warpPerspective(slika, H, (canvas_w, canvas_h))
        maska = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        if maska.sum() == 0:
            continue
        dist = cv.distanceTransform(maska, cv.DIST_L2, 5).astype(np.float32)
        if dist.max() > 0:
            dist = dist / (dist.max() + 1e-9)
        else:
            dist = maska.astype(np.float32) / 255.0

        accum += warped.astype(np.float32) * dist[:, :, None]
        weight += dist

    weight3 = weight[:, :, None]
    weight3[weight3 == 0] = 1.0
    rezultat = (accum / weight3).astype(np.uint8)
    return rezultat



def spoji_3_slike(slika_levo, slika_sredina, slika_desno, feature_scale=1.0, blend_metoda='feather'):
    slike = [slika_levo, slika_sredina, slika_desno]
    Hs = [None] * 3
    Hs[1] = np.eye(3, dtype=np.float64)

    # left -> mid
    tacke_levo, tacke_levo_sredina = detekcija_i_matchovanje(slika_levo, slika_sredina, scale=feature_scale)
    if tacke_levo is None:
        raise RuntimeError("Nema dovoljno pogodaka izmedju sredine i leve strane")
    H_levo_sredina, mask_l = izracunaj_homografiju(tacke_levo, tacke_levo_sredina)
    if H_levo_sredina is None:
        raise RuntimeError("Nije moguce izracunati homografiju levo->sredina")
    Hs[0] = H_levo_sredina

    # right -> mid
    tacke_desno, tacke_desno_sredina = detekcija_i_matchovanje(slika_desno, slika_sredina, scale=feature_scale)
    if tacke_desno is None:
        raise RuntimeError("Nema dovoljno pogodaka izmedju sredine i desne strane")
    H_desno_sredina, mask_r = izracunaj_homografiju(tacke_desno, tacke_desno_sredina)
    if H_desno_sredina is None:
        raise RuntimeError("Nije moguce izracunati homografiju desno->sredina")
    Hs[2] = H_desno_sredina

    H_canvas, canvas_size, offset = priprema_kanvasa_i_transformacija(slike, Hs)

    if blend_metoda == 'average':
        rezultat = preslikaj_avg_blend(slike, H_canvas, canvas_size)
    elif blend_metoda == 'feather':
        rezultat = preslikaj_feather_blend(slike, H_canvas, canvas_size)
    else:
        raise ValueError("Unknown blend_method: choose 'average', 'feather' or 'pyramid'")

    return rezultat

if __name__ == "__main__":
    # load images (BGR)
    slika_levo = cv.imread("1.jpg")
    slika_sredina = cv.imread("2.jpg")
    slika_desno = cv.imread("3.jpg")

    if slika_levo is None or slika_sredina is None or slika_desno is None:
        print("Check image paths (1.jpg, 2.jpg, 3.jpg).")
        exit(1)

    # OPCIJE:
    FEATURE_SCALE = 1     # 1.0 = full resolution detection; <1.0 = downscale detection (faster)
    BLEND_METODA = 'feather'   # 'average' | 'feather'

    # Run stitching
    rezultat = spoji_3_slike(slika_levo, slika_sredina, slika_desno,
                             feature_scale=FEATURE_SCALE,
                             blend_metoda=BLEND_METODA)

    cv.imwrite("panorama_three.png", rezultat)
    cv.imshow("Panorama", rezultat)
    cv.waitKey(0)
    cv.destroyAllWindows()
