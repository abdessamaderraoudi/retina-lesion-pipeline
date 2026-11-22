import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold, frangi, threshold_otsu
from skimage.measure import label, regionprops

DEFAULTS = dict(
    resize=1024, fov_thr=15, fov_erode=8,
    illum_method="median", illum_size=51,
    denoise="gaussian", denoise_size=3,
    clahe_clip=2.0, clahe_tile=8,
    lesion="dark", se_shape="ellipse", se_size=9,
    vessel_excl=False, vessel_sigma=3, vessel_pct=90,
    seg_method="otsu", offset=0, thr=20, block=51, adapt_c=5,
    hyst_low=10, hyst_high=30,
    min_area=5, max_area=2000, min_circ=0.0,
)
SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 1. Prétraitement
def fov_mask(img, thr, erode):
    """Masque circulaire du champ de vue (plus grande composante, trous remplis, érodée)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (gray > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    if erode > 0:
        filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2))
    return filled.astype(bool)


def preprocess(img, fov, p):
    """Canal vert -> correction d'illumination -> débruitage -> CLAHE. Retourne les images intermédiaires."""
    g = img[:, :, 1].copy()
    mean_in = float(g[fov].mean()) if fov.any() else float(g.mean())
    g[~fov] = int(mean_in)
    out = {"green": g}
    if p["illum_method"] != "aucune":
        k = int(p["illum_size"])
        bg = cv2.medianBlur(g, k) if p["illum_method"] == "median" else cv2.GaussianBlur(g, (0, 0), k / 3)
        g = np.clip(g.astype(np.float32) - bg.astype(np.float32) + mean_in, 0, 255).astype(np.uint8)
    out["corrected"] = g
    k = int(p["denoise_size"])
    if p["denoise"] == "gaussian":
        g = cv2.GaussianBlur(g, (k, k), 0)
    elif p["denoise"] == "median":
        g = cv2.medianBlur(g, k)
    elif p["denoise"] == "bilateral":
        g = cv2.bilateralFilter(g, k, 50, 2 * k)
    out["denoised"] = g
    if p["clahe_clip"] > 0:
        g = cv2.createCLAHE(clipLimit=float(p["clahe_clip"]),
                            tileGridSize=(int(p["clahe_tile"]),) * 2).apply(g)
    out["enhanced"] = g
    return out


# ---------------------------------------------------------------- 2. Morphologie
def morphology(enh, fov, p):
    """Black-hat (lésions sombres) ou top-hat (lésions claires) + exclusion optionnelle des vaisseaux (Frangi)."""
    se = cv2.getStructuringElement(SHAPES[p["se_shape"]], (int(p["se_size"]),) * 2)
    op = cv2.MORPH_BLACKHAT if p["lesion"] == "dark" else cv2.MORPH_TOPHAT
    resp = cv2.morphologyEx(enh, op, se)
    resp[~fov] = 0
    vessels = None
    if p["vessel_excl"]:
        f = frangi(enh.astype(np.float32) / 255, sigmas=range(1, int(p["vessel_sigma"]) + 1), black_ridges=True)
        t = np.percentile(f[fov], p["vessel_pct"])
        vessels = cv2.dilate(((f >= t) & fov).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        resp[vessels] = 0
    return resp, vessels


# ---------------------------------------------------------------- 3. Segmentation
def segment(resp, fov, p):
    m = p["seg_method"]
    if m == "otsu":
        b = resp > threshold_otsu(resp[fov]) + p["offset"]
    elif m == "global":
        b = resp > p["thr"]
    elif m == "adaptatif":
        b = cv2.adaptiveThreshold(resp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                  int(p["block"]), -float(p["adapt_c"])) > 0
    else:  # hystérésis
        b = apply_hysteresis_threshold(resp, p["hyst_low"], p["hyst_high"])
    return postprocess(b & fov, p)


def postprocess(b, p):
    """Filtrage des composantes par aire et circularité."""
    lab = label(b, connectivity=2)
    keep = np.zeros(b.shape, np.uint8)
    n = 0
    for r in regionprops(lab):
        if not (p["min_area"] <= r.area <= p["max_area"]):
            continue
        if p["min_circ"] > 0 and r.perimeter > 0 and 4 * np.pi * r.area / r.perimeter ** 2 < p["min_circ"]:
            continue
        keep[tuple(r.coords.T)] = 1
        n += 1
    return keep, n


# ---------------------------------------------------------------- Pipeline complet
def run_pipeline(img_bgr, params=None):
    p = {**DEFAULTS, **(params or {})}
    img = resize_max(img_bgr, int(p["resize"]))
    fov = fov_mask(img, p["fov_thr"], int(p["fov_erode"]))
    pre = preprocess(img, fov, p)
    resp, vessels = morphology(pre["enhanced"], fov, p)
    mask, n = segment(resp, fov, p) if fov.any() else (np.zeros(fov.shape, np.uint8), 0)
    return dict(img=img, fov=fov, **pre, response=resp, vessels=vessels, mask=mask, n_obj=n)


def overlay(img, mask, gt=None):
    """Contours du masque détecté (vert) et de la vérité terrain (orange) sur l'image."""
    out = img.copy()
    if gt is not None:
        c, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, c, -1, (0, 128, 255), 1)
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, c, -1, (0, 255, 0), 1)
    return out
