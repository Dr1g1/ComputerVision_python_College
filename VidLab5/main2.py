import cv2 as cv
import numpy as np
import glob
import os

# provera da li putanje postoje
FILES_PATH = 'files/*.jpg'
VIDEO_PATH = 'files/Aruco_board.mp4'

# KALIBRACIJA:
BOARD_SIZE = (5, 7)
MARKER_SIZE = 2
MARKER_SEPARATION = 0.4

aruco_recnik = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_1000)
parametri = cv.aruco.DetectorParameters()
tabla = cv.aruco.GridBoard(
    size = BOARD_SIZE,
    markerLength = MARKER_SIZE,
    markerSeparation = MARKER_SEPARATION,
    dictionary = aruco_recnik
)

detektor = cv.aruco.ArucoDetector(aruco_recnik, parametri)

def kalibracija_kamere(fajlovi_putanja: str, use_subpix = True):
    print("Pokrecem kalibraciju...")
    slike = glob.glob(fajlovi_putanja)

    if not slike:
        print(f"Greska: Nema slika na putanji {fajlovi_putanja}")
        return None

    svi_uglovi = []
    svi_idjevi = []
    brojac = []

    oblik_slike = None

    # parametri za cornerSubPix
    kriterijum = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    vel_prozora = (5, 5)
    mrtva_zona = (-1, -1)

    for f_ime in slike:
        slika = cv.imread(f_ime)
        if slika is None:
            continue

        gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
        oblik_slike = gray.shape

        uglovi, idjevi, rejected = detektor.detectMarkers(gray)

        if idjevi is not None and len(idjevi) > 0:
            # opcionalno refine svakog markera sa cornerSubPix
            if use_subpix:
                refined_list = []
                for c in uglovi:
                    # c ima oblik (4,1,2); reshape u (4,2)
                    pts = c.reshape(-1, 2).astype(np.float32)
                    # cornerSubPix ocekuje (N,2) ili (N,1,2)
                    pts_refined = cv.cornerSubPix(gray, pts, vel_prozora, mrtva_zona, kriterijum)
                    # vrati oblik (4,1,2)
                    refined_list.append(pts_refined.reshape(-1, 1, 2))
                uglovi = refined_list

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

    brojac_niz = np.array(brojac, dtype=np.int32)
    svi_idjevi_niz = np.array(svi_idjevi, dtype=np.int32)

    print(f"Ukupno markera: {len(svi_idjevi_niz)}")
    print(f"Broj frejmova (brojac): {len(brojac_niz)}")

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

    h, w = oblik_slike
    new_matrica_kamere, regija_od_interesa = cv.getOptimalNewCameraMatrix(matrica, distort, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(matrica, distort, None, new_matrica_kamere, (w, h), 5)

    print("Pokrećem video obradu... Pritisni 'q' za izlaz.")

    while True:
        ret, frame = video_fajl.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, idjevi, rejected = detektor.detectMarkers(gray)
        frame_output = frame.copy()

        if idjevi is not None and len(idjevi) > 0:
            cv.aruco.drawDetectedMarkers(frame_output, corners, idjevi, (0, 255, 0))

            objPoints, imgPoints = tabla.matchImagePoints(corners, idjevi)

            if len(objPoints) > 0:
                retval, vektor_rotacije, vektor_translacije = cv.solvePnP(objPoints, imgPoints, matrica, distort)

                if retval:
                    cv.drawFrameAxes(frame_output, matrica, distort, vektor_rotacije, vektor_translacije, MARKER_SIZE * 1.5)

        frame_bez_distorzije = cv.remap(frame_output, mapx, mapy, cv.INTER_LINEAR)

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
    if glob.glob(FILES_PATH):
        calib_data = kalibracija_kamere(FILES_PATH)

        if calib_data:
            matrix, dist_coeffs, shape = calib_data
            procena_pozicije_fja(VIDEO_PATH, matrix, dist_coeffs, shape)
    else:
        print("Nisu pronadjene slike za kalibraciju. Proveri putanju 'files/*.jpg'")

