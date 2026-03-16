import cv2 as cv
import numpy as np

# deskriptor - vektor brojeva koji opisuje kako izgleda okolina jedne karakteristicne tacke
# keypoints - karakteristicne tacke; svaka tacka sadrzi koordinate i info o uglu i velicini okruzenja

# detekcija karakteristicnih tacaka i match-ovanje
# David Lowe je empirijski dokazao da je 0.7 najbolja vrednost jer nema previse matcheva i dovoljno je jasna razlika
def detekcija_i_matchovanje(slika1, slika2, ratio_thresh=0.7, scale=1.0):
    # ratio_thresh - Lowe-ov prag za odstupanje - koliko je najbolji match bolji od drugog najboljeg match-al odstupanje je u prostoru deskriptora
    # scale - faktor smanjenja pri detekciji - ako je scale < 1 onda je detekcija brza sa identicnim rezultatima jer se znacajno smanjuje broj piksela i keypointa
    if scale != 1.0:
        skalirana_slika1 = cv.resize(slika1, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        skalirana_slika2 = cv.resize(slika2, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    else:
        skalirana_slika1, skalirana_slika2 = slika1, slika2

    # sift detekcija i isracunavanje deskriptora
    sift = cv.SIFT_create() # instanciramo SIFT objekat
    ktacke1, deskriptori1 = sift.detectAndCompute(skalirana_slika1, None)
    ktacke2, deskriptori2 = sift.detectAndCompute(skalirana_slika2, None)

    if deskriptori1 is None or deskriptori2 is None:
        return None, None # ako algoritam nije pronasao nista

    # KD-tree -> K dimensional tree - struktura podataka
    # FLANN matcher koji koristi KD-tree; Fast Library for Approximate Nearest Neighbors
    # flann je automatski optimizovan - dizajniran da sam bira najbolje parametre radi brzine
    # ovde uparujemo tacke - trazenje parova
    # ovde se koristi KD-tree algoritam; trees=5 znaci da ce algoritam da napravi 5 paralelnih stabala pretrage; vise stabala - veca preciznost i vise potrosene memorije
    FLANN_INDEX_KDTREE = 1
    indeks_parametri = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    parametri_trazenja = dict(checks=50) # checks je koliko puta ce algoritam proci kroz stabla trazeci najbolji par
    flann = cv.FlannBasedMatcher(indeks_parametri, parametri_trazenja)

    mecovi = flann.knnMatch(deskriptori1, deskriptori2, k=2) # nadji 2 najbolja pogotka za svaku tacku
    tacke1 = []
    tacke2 = []

    # sad kad smo nasli gomilu parova treba da prodju Lowe-ov Ratio test
    # ovaj test uporedjuje stepen slicnosti(matematicku distancu) - uporedjuje deskriptore
    for m_n in mecovi:
        if len(m_n) < 2:
            continue
        m, n = m_n # za svaku tacku gledamo 2 najbolja pogotka; ovde koristimo unpacking u pajtonu
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


# Funkcija koja racuna homografiju:
# homografija - matematicka matrica koja opisuje kako se jedna ravan(slika) transformise u drugu
# koristi se ransac algoritam(random sample consensus)
# ransac_thresh - prag u pikselima - koliko projekcija moze da se omasi a da se tacka i dalje smatra inlierom(tacka koja se uklapa u izabrani matematicki model a model je homografija)
# homografija je matematicki opis kako se jedna ravan preslikava u drugu - kako da se ispruzi, nagne, rotira i pomeri jedna slika da legne preko druge
def izracunaj_homografiju(tacke_src, tacke_dst, ransac_thresh=5.0):
    # tacke_src - tacke sa izvorne slike; tacke_dst - odgovarajuce tacke u ciljnoj slici
    # tacke_src[i] i tacke_dst[i] opisuje istu FIZICKU tacku
    if tacke_src is None or tacke_dst is None:
        return None, None
    if len(tacke_src) < 4 or len(tacke_dst) < 4:
        return None, None
    H, maska = cv.findHomography(tacke_src, tacke_dst, cv.RANSAC, ransac_thresh)
    return H, maska # vraca H - homografiju i masku; maska nam govori koji mecevi su inlieri a koji su outlieri


# Canvas / homographies assembly (mapira svaku sliku prema koordinatama srednje slike)
def priprema_kanvasa_i_transformacija(slike, homografije):
    svi_uglovi = [] #svi uglovi svih slika
    for slika, H in zip(slike, homografije):
        h, w = slika.shape[:2] # trazimo visinu i sirinu svake slike
        uglovi = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32).reshape(-1, 1, 2)
        if H is None:
            tacke = uglovi # ako je referentna slika onda njeni uglovi ostaju gde su bili
        else:
            tacke = cv.perspectiveTransform(uglovi, H)
        svi_uglovi.append(tacke)

    svi_uglovi = np.concatenate(svi_uglovi, axis=0) # spajanje svih tacaka
    x_min, y_min = np.int32(svi_uglovi.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(svi_uglovi.max(axis=0).ravel() + 0.5) # pronalazimo granice panorame da bismo znali koliki canvas nam treba

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0 # isracunavamo pomeraj

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64) # translaciona matrica

    H_canvas = [(T @ H).astype(np.float64) for H in homografije]

    return H_canvas, (canvas_h, canvas_w), (tx, ty) # vracamo homografije spremne za warpPerspective, dimenzije panorame i pomeraj


# preslikava slike na kanvas
def preslikaj_avg_blend(slike, homografije, velicina_kanvasa):
    canvas_h, canvas_w = velicina_kanvasa
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for slika, H in zip(slike, homografije):
        warped = cv.warpPerspective(slika, H, (canvas_w, canvas_h)) # geometrijsko preslikavanja - slika se projektuje na canvas; svi pikseli idu na svoje globalne koordinate
        maska = (warped.sum(axis=2) > 0).astype(np.float32) # pravimo masku za validne piksele - imacemo sliku i pozadinu koja ce da bude crna
        accum += warped.astype(np.float32) # akumulator boja
        weight += maska # akumulator slika

    weight3 = weight[:, :, None]
    weight3[weight3 == 0] = 1.0
    rezultat = (accum / weight3).astype(np.uint8) # racunanje proseka
    return rezultat

# Feather blending - postepeno se gasi uticaj slike ka njenim ivicama
# znaci distance transform ovde sluzi da bi pikselima blizim centru slike dodelio vece tezine, a pikselima blizim ivicama manje - bolji prelaz
def preslikaj_feather_blend(slike, homografije, vel_kanvas):
    canvas_h, canvas_w = vel_kanvas
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32) # zbir boja
    #Akumulator accum sabira sve RGB vrednosti piksela, pomnožene sa njihovim težinama (distance transform za feather, maska 0/1 za average).
    # Svaki piksel u accum = ukupan doprinos svih slika u tom pikselu
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32) # zbir tezina

    for slika, H in zip(slike, homografije): # svaka slika se posebno preslikava
        warped = cv.warpPerspective(slika, H, (canvas_w, canvas_h))
        maska = (warped.sum(axis=2) > 0).astype(np.uint8) * 255 # pravimo masku
        if maska.sum() == 0:
            continue
        dist = cv.distanceTransform(maska, cv.DIST_L2, 5).astype(np.float32)
        # ovo iznad - racuna se udaljenost do najblize ivice(pozadine)
        # normalizuj na [0,1]
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
