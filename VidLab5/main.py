import cv2 as cv
import numpy as np
import glob
import os

# provera da li putanje postoje
FILES_PATH = 'files/*.jpg'
VIDEO_PATH = 'files/Aruco_board.mp4'

# KALIBRACIJA:
BOARD_SIZE = (5, 7)  # broj markera (width, height)
MARKER_SIZE = 2  # velicina markera (npr. cm ili m) - bitno za translaciju
MARKER_SEPARATION = 0.4

# ARUCO PODESAVANJA:
# recnik binarnih kodova koji definisu markere - 6x6 bitova, 1000 mogucih kombinacija
aruco_recnik = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_1000)
# kreiranje parametara:
parametri = cv.aruco.DetectorParameters()
# kreiranje table:
tabla = cv.aruco.GridBoard( # ocekuj ove markere(recnik) poredjane u 5 redova i 7 kolona, gde je svaki marker velicine 2 jedinice, a razmak izmedju njih je 0.4 jedinice
    # ovo nam omogucava da definisemo 3D koordinate celog papira
    size = BOARD_SIZE,
    markerLength = MARKER_SIZE,
    markerSeparation = MARKER_SEPARATION,
    dictionary = aruco_recnik
)
# kreiranje detektora - detektor se inicijalizuje jednom jer je tako optimalnije
detektor = cv.aruco.ArucoDetector(aruco_recnik, parametri)
# detektor je objekat koji sadrzi logiku za prepoznavanje crno-belih kvadrata na slici

# treba da nadjemo intrinsic parametre kamere(fokusna duzina i opticki centar) i koeficijente distorzije
def kalibracija_kamere(fajlovi_putanja: str):
    print("Pokrecem kalibraciju...")
    slike = glob.glob(fajlovi_putanja) # ucitavamo sve slike

    # ako nije nasao slike nikakve
    if not slike:
        print(f"Greska: Nema slika na putanji {fajlovi_putanja}")
        return None

    # liste za cuvanje podataka
    # cuvamo sve pronadjene uglove i ID-eve u liste
    svi_uglovi = [] # lista 2d koordinata uglova markera detektovanih na slikama
    svi_idjevi = [] # id brojevi svakog markera koji omogucavaju fji da upari 2d tacku sa slike sa fiksnom 3d tackom na modelu table
    brojac = []  # ovde brojimo koliko markera ima na svakoj slici

    oblik_slike = None

    for f_ime in slike:
        slika = cv.imread(f_ime)
        if slika is None: continue

        gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
        oblik_slike = gray.shape # za svaku sliku se vrsi konverzija u sivi mod jer lagoritmi za
        # detekciju ivica i markera rade iskljucivo sa intenzitetom svetlosti a ne bojama

        # detekcija
        uglovi, idjevi, rejected = detektor.detectMarkers(gray) # trazi kvadratne sare aruco recnika na slici

        if idjevi is not None and len(idjevi) > 0:
            # dodajemo broj pronadjenih markera u ovom frejmu u brojac
            brojac.append(len(idjevi))

            # dodajemo pronadjene markere i id-jeve u dugacke liste
            svi_uglovi.extend(uglovi)

            # ids je numpy array - konvertujemo ga u listu ili extendujemo
            for marker_id in idjevi:
                svi_idjevi.append(marker_id)

    if not svi_uglovi:
        print("Nisu pronadjeni markeri na slikama za kalibraciju.")
        return None

    # konverzija u numpy nizova za opencv
    # counter mora biti numpy niz intova
    brojac_niz = np.array(brojac, dtype=np.int32)
    # ids mora biti numpy niz
    svi_idjevi_niz = np.array(svi_idjevi, dtype=np.int32)

    print(f"Ukupno markera: {len(svi_idjevi_niz)}")
    print(f"Broj frejmova (brojac): {len(brojac_niz)}")

    # kamera_mat - sadrzi fokusnu daljinu(fx, fy) i opticki centar(cx, cy)
    # distort_koef - opisuju kako socivo krivi sliku
    # reprojection_greska - broj koji predstavlja prosecnu gresku u pikselima - idealno je da je manje od 1.0 jer je kalibracija preciznija
    # ovo je potrebno jer iako je distorzija zanemarljiva - treba nam maksimalna preciznost; ose koje iscrtavam na videu ne bi pratile tablu kako treba inace
    try:
        reprojection_greska, kamera_mat, distort_koef, rvecs, tvecs = cv.aruco.calibrateCameraAruco(
            svi_uglovi,
            svi_idjevi_niz,
            brojac_niz,
            tabla,
            oblik_slike[::-1],
            None,
            None
        )
        print(f"Kalibracija uspesna. Reprojection error: {reprojection_greska}")
        return kamera_mat, distort_koef, oblik_slike

    except cv.error as e:
        print(f"Greska tokom izvrsavanja calibrateCameraAruco f-je: {e}")
        return None


def procena_pozicije_fja(video_putanja, matrica, distort, oblik_slike):
    video_fajl = cv.VideoCapture(video_putanja)
    if not video_fajl.isOpened():
        print("Greska pri otvaranju videa.")
        return

    # OPTIMIZACIJA: pre-kalkulacija mapa za undistort
    # Ovo radimo SAMO JEDNOM pre petlje
    h, w = oblik_slike
    # izracunava novu matricu kamere koja nam omogucava da kontrolisemo sta se desava sa ivicama slike nakon sto je ispeglamo od distorzije
    new_matrica_kamere, regija_od_interesa = cv.getOptimalNewCameraMatrix(matrica, distort, (w, h), 1, (w, h))
    # pravimo mapu transformacije - za piksel na poziciji 1 na novoj slici idi na tu i tu kordinatu na staroj slici
    mapx, mapy = cv.initUndistortRectifyMap(matrica, distort, None, new_matrica_kamere, (w, h), 5)

    print("Pokrećem video obradu... Pritisni 'q' za izlaz.")

    while True:
        ret, frame = video_fajl.read()
        if not ret:
            break

        # detekcija markera na ORIGINALNOM frejmu (pre undistort-a)
        # veoma je vazno detektovati na originalnoj slici jer matrica i distort
        # odgovaraju toj distorziji.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, idjevi, rejected = detektor.detectMarkers(gray)
        # detekcija se radi na originalnom frejmu - jer su parametri kalibracije za izracunati sa distorzjom
        # ako bismo na slici bez distorzije trazili markere ne bi bilo dobro
        frame_output = frame.copy()

        if idjevi is not None and len(idjevi) > 0:
            # crtanje detektovanih markera
            cv.aruco.drawDetectedMarkers(frame_output, corners, idjevi, (0, 255, 0))

            # ova funkcija poredi uglove koje smo upravo detektovali na slici sa
            # poznatim 3D koordinatama tih istih markera na papiru (koje smo definisali preko BOARD_SIZE i MARKER_SIZE).
            objPoints, imgPoints = tabla.matchImagePoints(corners, idjevi)

            if len(objPoints) > 0:
                # resavanje pnp problema - pronalazenje polozaja objekta ako znamo njegove 3d koordinate i njihove 2d projekcije na slici
                retval, vektor_rotacije, vektor_translacije = cv.solvePnP(objPoints, imgPoints, matrica, distort)

                if retval:
                    # crtanje osa koord. sistema
                    cv.drawFrameAxes(frame_output, matrica, distort, vektor_rotacije, vektor_translacije, MARKER_SIZE * 1.5)

        # uklanjanje distorzije
        # koristimo remap umesto undistort
        frame_bez_distorzije = cv.remap(frame_output, mapx, mapy, cv.INTER_LINEAR)

        # resize za prikaz - da stane na ekran
        scale_percent = 60
        width = int(frame_bez_distorzije.shape[1] * scale_percent / 100)
        height = int(frame_bez_distorzije.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_show = cv.resize(frame_bez_distorzije, dim, interpolation=cv.INTER_AREA)

        cv.imshow("Aruco Pose Estimation", resized_show)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_fajl.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # provera da li fajlovi postoje pre pokretanja:
    if glob.glob(FILES_PATH):
        calib_data = kalibracija_kamere(FILES_PATH)

        if calib_data:
            matrix, dist_coeffs, shape = calib_data
            procena_pozicije_fja(VIDEO_PATH, matrix, dist_coeffs, shape)
    else:
        print("Nisu pronadjene slike za kalibraciju. Proveri putanju 'files/*.jpg'")



# kalibracija_kamere
# Ova funkcija analizira set fotografija ArUco table kako bi precizno izračunala
# unutrašnje parametre sočiva i koeficijente krivljenja (distorzije) kamere.
# Na osnovu detektovanih markera, ona generiše matricu kamere koja je neophodna
# za tačno postavljanje 3D objekata i osa u prostoru.
#
# procena_pozicije_fja Ova funkcija u realnom vremenu obrađuje video snimak,
# identifikuje tablu i izračunava njenu tačnu rotaciju i udaljenost u odnosu na
# kameru. Koristeći prethodno izračunatu mapu transformacije, ona ispravlja optička
# iskrivljenja slike i iscrtava koordinatne ose direktno na video frejmu.

# Tok je ovakav:
# Znaš 3D koordinate uglova šahovske table (idealne, u realnom svetu)
# Znaš 2D koordinate tih uglova u slici (detektovane iz frejmova)
# Nakon kalibracije imaš:
# parametre kamere (fokus, centar)
# koeficijente distorzije
# Sada:
# uzmeš 3D tačke
# pomoću dobijenih parametara ih projektuješ nazad u 2D sliku
# uporediš ih sa stvarno detektovanim 2D uglovima
# Razlika između ta dva položaja = reprojection error