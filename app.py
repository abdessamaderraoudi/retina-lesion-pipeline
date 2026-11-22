"""G10 – Exploration interactive du pipeline de détection de lésions sur fond d'œil.
Lancer :  streamlit run app.py"""
import json

import cv2
import numpy as np
import streamlit as st

from metrics import compute_metrics, curves, fit_gt, sweep
from pipeline import DEFAULTS, overlay, run_pipeline

st.set_page_config(page_title="G10 – Lésions fond d'œil", layout="wide")
st.title("Détection de lésions sur fond d'œil – pipeline interactif")


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def cached_run(data, params_json):
    return run_pipeline(decode(data), json.loads(params_json))


def show(img, caption, **kw):
    st.image(img, caption=caption, channels="BGR" if img.ndim == 3 else "GRAY", clamp=True, **kw)


def norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)


# ------------------------------------------------------------------ Entrées
up = st.sidebar.file_uploader("Image de fond d'œil", type=["png", "jpg", "jpeg", "tif", "tiff"])
gt_up = st.sidebar.file_uploader("Vérité terrain (masque binaire, optionnel)", type=["png", "jpg", "tif", "tiff"])
if up is None:
    st.info("Importe une image de fond d'œil (ex. IDRiD, DIARETDB1, e-ophtha) dans la barre latérale.")
    st.stop()
data = up.getvalue()
gt_raw = cv2.imdecode(np.frombuffer(gt_up.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE) if gt_up else None

# ------------------------------------------------------------------ Paramètres par étape
P = dict(DEFAULTS)
sb = st.sidebar
P["resize"] = sb.select_slider("Taille max (px)", [512, 768, 1024, 1536], 1024)
with sb.expander("1. Prétraitement", expanded=True):
    P["illum_method"] = st.selectbox("Correction d'illumination", ["median", "gaussian", "aucune"])
    P["illum_size"] = st.slider("Taille du fond estimé", 11, 201, 51, 2)
    P["denoise"] = st.selectbox("Débruitage", ["gaussian", "median", "bilateral", "aucun"])
    P["denoise_size"] = st.slider("Taille du filtre", 3, 15, 3, 2)
    P["clahe_clip"] = st.slider("CLAHE clipLimit (0 = off)", 0.0, 10.0, 2.0, 0.1)
    P["clahe_tile"] = st.slider("CLAHE tileGridSize", 2, 32, 8)
    P["fov_thr"] = st.slider("Seuil FOV", 1, 60, 15)
    P["fov_erode"] = st.slider("Érosion FOV (px)", 0, 30, 8)
with sb.expander("2. Morphologie", expanded=True):
    P["lesion"] = st.radio("Cible", ["dark", "bright"], format_func=lambda x: {
        "dark": "Rouges (microanévrismes/hémorragies) – black-hat",
        "bright": "Claires (exsudats) – top-hat"}[x])
    P["se_shape"] = st.selectbox("Forme de l'élément structurant", ["ellipse", "rect", "cross"])
    P["se_size"] = st.slider("Taille de l'élément structurant (finesse)", 1, 51, 9, 2)
    P["vessel_excl"] = st.checkbox("Exclure les vaisseaux (Frangi)")
    if P["vessel_excl"]:
        P["vessel_sigma"] = st.slider("σ max Frangi", 1, 6, 3)
        P["vessel_pct"] = st.slider("Percentile vaisseaux", 80.0, 99.5, 90.0, 0.5)
with sb.expander("3. Segmentation", expanded=True):
    P["seg_method"] = st.selectbox("Méthode", ["otsu", "global", "adaptatif", "hysteresis"])
    if P["seg_method"] == "otsu":
        P["offset"] = st.slider("Décalage du seuil d'Otsu", -30, 60, 0)
    elif P["seg_method"] == "global":
        P["thr"] = st.slider("Seuil", 1, 100, 20)
    elif P["seg_method"] == "adaptatif":
        P["block"] = st.slider("Taille du bloc", 3, 151, 51, 2)
        P["adapt_c"] = st.slider("Constante C", -10, 40, 5)
    else:
        P["hyst_low"], P["hyst_high"] = st.slider("Seuils bas / haut", 1, 120, (10, 30))
    P["min_area"] = st.slider("Aire min (px)", 1, 100, 5)
    P["max_area"] = st.slider("Aire max (px)", 50, 20000, 2000, 50)
    P["min_circ"] = st.slider("Circularité min", 0.0, 1.0, 0.0, 0.05)

with st.spinner("Calcul du pipeline…"):
    R = cached_run(data, json.dumps(P))
gt = fit_gt(gt_raw, R["mask"].shape) if gt_raw is not None else None

# ------------------------------------------------------------------ Affichage
t1, t2, t3, t4, t5 = st.tabs(["Prétraitement", "Morphologie", "Segmentation", "Évaluation", "Analyse de sensibilité"])
with t1:
    c = st.columns(4)
    for col, (k, cap) in zip(c, [("green", "Canal vert"), ("corrected", "Illumination corrigée"),
                                 ("denoised", "Débruitée"), ("enhanced", "CLAHE")]):
        with col:
            show(R[k], cap, use_container_width=True)
    st.caption("Masque FOV : %d px" % R["fov"].sum())
with t2:
    c = st.columns(2 if R["vessels"] is not None else 1)
    with c[0]:
        show(norm(R["response"]), "Réponse top-hat / black-hat (normalisée)", use_container_width=True)
    if R["vessels"] is not None:
        with c[1]:
            show(R["vessels"].astype(np.uint8) * 255, "Vaisseaux exclus (Frangi)", use_container_width=True)
with t3:
    c = st.columns(2)
    with c[0]:
        show(R["mask"] * 255, "Masque binaire", use_container_width=True)
    with c[1]:
        show(overlay(R["img"], R["mask"], gt), "Superposition (vert : détection, orange : vérité terrain)",
             use_container_width=True)
    m1, m2 = st.columns(2)
    m1.metric("Objets détectés", R["n_obj"])
    m2.metric("Aire détectée (% du FOV)", f"{100 * R['mask'][R['fov']].mean():.3f}")
with t4:
    if gt is None:
        st.info("Importe un masque de vérité terrain pour obtenir sensibilité, spécificité, précision, Dice/IoU et courbes.")
    else:
        M = compute_metrics(R["mask"], gt, R["fov"])
        cols = st.columns(5)
        for col, k in zip(cols, ["sensibilite", "specificite", "precision", "dice_f1", "iou"]):
            col.metric(k, f"{M[k]:.3f}")
        st.caption(f"TP={M['tp']}  FP={M['fp']}  FN={M['fn']}")
        cv = curves(R["response"], gt, R["fov"])
        if cv:
            a, b = st.columns(2)
            a.subheader(f"ROC (AUC = {cv['auc']:.3f})")
            a.line_chart(cv["roc"], x="fpr", y="tpr")
            b.subheader(f"Précision-rappel (AP = {cv['ap']:.3f})")
            b.line_chart(cv["pr"], x="rappel", y="precision")
with t5:
    st.write("Fait varier **un** paramètre en gardant les autres réglages courants.")
    opts = {"clahe_clip": ("CLAHE clipLimit", 0.0, 10.0, 0.5), "clahe_tile": ("CLAHE tileGridSize", 2, 32, 4),
            "se_size": ("Taille de l'élément structurant", 1, 41, 4), "offset": ("Décalage Otsu", -20, 60, 10),
            "thr": ("Seuil global", 1, 100, 10), "min_area": ("Aire min", 1, 100, 10),
            "denoise_size": ("Taille du débruitage", 3, 15, 4), "illum_size": ("Taille du fond", 11, 201, 6)}
    key = st.selectbox("Paramètre", list(opts), format_func=lambda k: opts[k][0])
    _, lo, hi, _ = opts[key]
    a, b, n = st.columns(3)
    v0 = a.number_input("Début", value=float(lo))
    v1 = b.number_input("Fin", value=float(hi))
    n = n.number_input("Nb de points", 3, 20, 8)
    if st.button("Lancer l'analyse"):
        vals = np.linspace(v0, v1, int(n))
        if key in ("clahe_tile", "min_area", "offset", "thr"):
            vals = np.unique(np.round(vals).astype(int))
        elif key in ("se_size", "denoise_size", "illum_size"):  # tailles impaires
            vals = np.unique((np.round(vals).astype(int) // 2) * 2 + 1)
        with st.spinner("Balayage…"):
            df = sweep(decode(data), P, key, vals.tolist(), gt_raw)
        st.dataframe(df.style.format("{:.3f}"))
        st.line_chart(df[[c for c in ["sensibilite", "specificite", "precision", "dice_f1", "iou"] if c in df]]
                      if gt is not None else df[["n_objets"]])
        st.download_button("Exporter en CSV", df.to_csv().encode(), f"sweep_{key}.csv")
