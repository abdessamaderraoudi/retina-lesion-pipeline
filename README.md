
## Installation 
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

## Structure
- `pipeline.py` : prétraitement (canal vert, illumination, débruitage, CLAHE, FOV), morphologie
  (top/black-hat, élément structurant, Frangi), segmentation (Otsu, global, adaptatif, hystérésis,
  filtre aire/circularité). Une fonction par étape + `run_pipeline`.
- `metrics.py` : sensibilité, spécificité, précision, Dice/F1, IoU, ROC/PR, balayage de paramètre.
- `app.py` : interface Streamlit (sliders par étape, images intermédiaires, évaluation, sensibilité).

## Données
IDRiD (segmentation : MA, HE, EX, SE, masques .tif) ou DIARETDB1. Importer **un** masque binaire
par lésion (ex. microanévrismes) correspondant à l'image, puis choisir la cible (rouge/claire).

## Expériences conseillées pour le rapport
1. F1 vs `clipLimit` CLAHE (0 → 10) : trop fort = bruit amplifié, faux positifs.
2. F1 vs taille de l'élément structurant : petit (3-9) = microanévrismes, grand (15-41) = hémorragies/exsudats.
3. Avec/sans exclusion des vaisseaux (Frangi) : effet sur la précision.
4. Comparaison Otsu / adaptatif / hystérésis sur images bien et mal éclairées.
5. Courbes ROC/PR de la réponse morphologique selon la taille de l'élément structurant.
