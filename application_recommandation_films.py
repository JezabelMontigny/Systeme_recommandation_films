import pandas as pd
import re
import numpy as np
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Définir la configuration de la page
st.set_page_config(
    page_title="Decodex - Système de recommandations",
    page_icon="🎬",
    layout="wide",  # Mettre "wide" pour étendre la page
    initial_sidebar_state="expanded"  # La sidebar sera toujours affichée au départ
)

# Lien partie 1
url = "https://drive.google.com/file/d/1d667eGYQCc_LGMiX3OJhhMiuFyiKCzih/view?usp=drive_link"

# Regex pour récupérer le lien URL
pattern = r"https://drive.google.com/file/d/([a-zA-Z0-9_-]+)/view\?usp=drive_link"

# Utilisation de re.search pour trouver l'ID du fichier
match = re.search(pattern, url)

file_id = match.group(1)
# Créer l'URL de téléchargement direct avec l'ID extrait
download_url = f"https://drive.google.com/uc?id={file_id}"

# Charger le fichier CSV depuis l'URL de téléchargement direct
def charger_donnees():
    return pd.read_csv(download_url, delimiter="\t")

df = charger_donnees()

# URL de base pour les affiches
url_image_base = "https://image.tmdb.org/t/p/w600_and_h900_bestv2"

# Clé API TMDB
api_key = os.getenv("TMDB_API_KEY")

# Créer la colonne "Lien" en concaténant l'URL de base et la colonne "Affiche"
df["Lien"] = url_image_base + df["Affiche"]

# Pondération des colonnes
poids = {
    "Note_moyenne": 5, "Nb_votants": 5, "Annee_sortie": 5, "Duree_minutes": 5,
    "Film": 5, "Court_metrage": 5,
    "Comedie": 10, "Documentaire": 10, "Famille": 10, "Romance": 10,
    "DIR_": 20, "ACT_": 25  # Poids pour les colonnes des directeurs et acteurs
}

# Appliquer les pondérations
for col in df.columns:
    if col in poids:  # Pondération des colonnes générales
        df[col] = df[col] * poids[col]
    elif col.startswith("DIR_"):  # Pondération des colonnes des directeurs
        df[col] = df[col] * poids["DIR_"]
    elif col.startswith("ACT_"):  # Pondération des colonnes des acteurs
        df[col] = df[col] * poids["ACT_"]

# Sélection des colonnes numériques pour la similarité
features = df.select_dtypes(include=[np.number])

# Calcul de la similarité cosinus
cos_sim = cosine_similarity(features)

# Fonction de recommandation
def recommander_films(titre, n):
    try:
        # Récupérer l'index du film
        film_index = df[df["Titre_original"] == titre].index[0]

        # Obtenir les similarités cosinus et trier
        similarites = cos_sim[film_index]
        indices_similaires = np.argsort(similarites)[-n-1:-1][::-1]  # Top-n films similaires

        # Retourner les titres et les distances
        return [
            (df.iloc[i]["Titre_original"], df.iloc[i]["Lien"], df.iloc[i]["ID_film"]) for i in indices_similaires
        ]
    except IndexError:
        return f"Corrige le titre de ton film !"

# Fonction pour formater la date au format "JJ/MM/AAAA"
def formater_date(date_str):
    try:
        # Convertir la chaîne de date en objet datetime
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Formater la date en "JJ/MM/AAAA"
        return date_obj.strftime("%d/%m/%Y")  # Format : 06/02/2019
    except ValueError:
        return date_str  # Retourner la date telle quelle si la conversion échoue

def obtenir_details_tmdb(film_id):
    try:
        if not api_key:
            raise ValueError("Clé API manquante")

        url = f"https://api.themoviedb.org/3/movie/{film_id}?api_key={api_key}&language=fr"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"Erreur TMDB (film): {response.status_code}")

        data = response.json()
        duree = data.get("runtime", "N/A")
        genres = [genre["name"] for genre in data.get("genres", [])]
        synopsis = data.get("overview", "Aucun synopsis disponible.")
        date_sortie = data.get("release_date", "N/A")
        date_sortie_formatee = formater_date(date_sortie)

        acteurs = obtenir_acteurs(film_id)
        video_url = obtenir_video_youtube(film_id)

        # Récupérer le nom du réalisateur
        equipe_url = f"https://api.themoviedb.org/3/movie/{film_id}/credits?api_key={api_key}&language=fr"
        equipe_response = requests.get(equipe_url)
        if equipe_response.status_code == 200:
            equipe_data = equipe_response.json()
            directeur = next((m["name"] for m in equipe_data["crew"] if m["job"] == "Director"), "Inconnu")
        else:
            directeur = "Inconnu"

        note = data.get("vote_average", 0.0)
        budget = data.get("budget", 0)
        revenu = data.get("revenue", 0)

        return duree, genres, synopsis, date_sortie_formatee, acteurs, video_url, directeur, note, budget, revenu

    except Exception as e:
        return (
            "N/A",                # duree
            [],                   # genres
            f"Erreur : {e}",      # synopsis
            "N/A",                # date_sortie
            [],                   # acteurs
            None,                 # video_url
            "Inconnu",            # directeur
            0.0,                  # note
            0,                    # budget
            0                     # revenu
        )



        # Récupérer la note du film
        note = data.get("vote_average", "N/A")  # Si pas de note, on met "N/A"
        
        # Récupérer le budget et les revenus
        budget = data.get("budget", 0)  # Le budget peut être nul, donc mettre 0 si indisponible
        revenu = data.get("revenue", 0)  # Le revenu peut aussi être nul, donc mettre 0 si indisponible
        
        return duree, genres, synopsis, date_sortie_formatee, acteurs, video_url, directeur, note, budget, revenu
    else:
        return "N/A", [], "Aucun détail disponible.", "N/A", [], None, "Inconnu"  # Ajouter "Inconnu" pour le directeur en cas d'erreur

# Fonction pour obtenir les acteurs du film via l'API TMDB
def obtenir_acteurs(film_id):
    url = f"https://api.themoviedb.org/3/movie/{film_id}/credits?api_key={api_key}&language=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        acteurs = [acteur["name"] for acteur in data.get("cast", [])][:5]  # Limiter à X acteurs principaux
        return acteurs
    return []  # Si aucun acteur n'est trouvé, retourner une liste vide

# Fonction pour obtenir l'URL de la vidéo YouTube
def obtenir_video_youtube(film_id):
    url = f"https://api.themoviedb.org/3/movie/{film_id}/videos?api_key={api_key}&language=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            # Utilisation de la première vidéo disponible
            video_key = data["results"][0]["key"]
            return f"https://www.youtube.com/watch?v={video_key}"
        else:
            return None  # Aucun vidéo trouvée
    return None  # Erreur lors de la récupération des vidéos

# Initialisation de la taille du texte avec un état par défaut
if "taille_texte" not in st.session_state:
    st.session_state.taille_texte = 16  # Valeur par défaut

# Barre de navigation à gauche avec modification du style de la barre de recherche
with st.sidebar:
    # Paramétrer la largeur bloquante de la sidebar
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 350px !important; 
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="https://cms-assets.webediamovies.pro/cdn-cgi/image/dpr=1,fit=crop,gravity=auto,metadata=none,quality=85,width=1000,height=192/production/4135/eae7ed9d409146d624a6b38923be9fc0.jpg" />
    </div>
    <div style="height: 20px;"></div>  <!-- Espace après le bloc -->
    """,
    unsafe_allow_html=True
    )


    with st.sidebar.expander("Choisissez vos genres", expanded=False):
        # Créer deux colonnes dans l'expander pour afficher les checkboxes
        col1, col2 = st.columns(2)

        # Afficher les checkboxes dans les colonnes
        with col1:
            show_comedie = st.checkbox("Comédie", value=True)
            show_documentaire = st.checkbox("Documentaire", value=True)
        with col2:
            show_famille = st.checkbox("Famille", value=True)
            show_romance = st.checkbox("Romance", value=True)

        # Créer un masque pour chaque genre
        masque = []
        if show_comedie:
            masque.append(df["Comedie"] !=0)
        if show_documentaire:
            masque.append(df["Documentaire"] !=0)
        if show_famille:
            masque.append(df["Famille"] !=0)
        if show_romance:
            masque.append(df["Romance"] !=0)

        # Appliquer le masque sur le dataframe : cela signifie que l'on filtre les lignes qui correspondent à l'un des genres sélectionnés
        if masque:
            filtre = masque[0]
            for m in masque[1:]:
                filtre = filtre | m  # Utiliser 'ou' logique pour combiner les filtres

            # Filtrer le dataframe selon le masque combiné
            df_filtre = df[filtre]

            # Créer une liste des titres correspondant au genre sélectionné
            liste_titres = df_filtre["Titre_original"].tolist()
        else:
            liste_titres = []  # Si aucun genre n'est sélectionné, la liste est vide

    # Afficher la liste des titres filtrés
    st.write(f"Films dans la base de données : {len(liste_titres)}")

    liste_titres.sort()  # Tri alphabétique de la liste
    # Vérifier le nombre de films dans la liste
    if len(liste_titres) == 0:
        st.error("Aucun film ne correspond aux filtres actuels.")
        st.stop()  # Arrêter l'exécution si aucun film n'est trouvé

    else:
        liste_titres.sort()  # Tri alphabétique de la liste
        # Sélectionner un film sans valeur par défaut
    if len(liste_titres) == 0:
        st.stop()  # Arrêter l'exécution si aucun film n'est dans la liste

    liste_titres.insert(0, "")
    titre_film = st.selectbox("**Entrez le titre du film :**", liste_titres)

    # Vérifier si un film a été sélectionné
    if titre_film == "":
        st.stop()  # Arrêter l'exécution si aucun film n'est sélectionné
    else:
        film_saisi = df[df["Titre_original"] == titre_film]
        with st.expander("Affiche du Film choisi", expanded=True):

            # Affichage de l'image du film (colonne 'lien' contenant l'URL de l'image)
            if titre_film:
                try:
                    image_url = df[df["Titre_original"] == titre_film]["Lien"].values[0]
                    st.markdown(f"<div style='text-align:center; font-size:{st.session_state.taille_texte}px;'></div>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <style>
                        .image-container {{
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100%;
                        }}
                        .image-bordure {{
                            border: 1px solid #000000;  /* Bordure de 1px de couleur noire */
                            padding: 0px;  /* Espacement interne pour que la bordure ne touche pas l'image */
                            border-radius: 10px;  /* Coins arrondis pour un effet plus doux */
                            max-height: 350px;  /* Hauteur maximale de l'image */
                            object-fit: cover;  /* L'image se redimensionne tout en conservant ses proportions */
                            margin-bottom: 10px;  /* Espace de 20px sous l'image */
                        }}
                        </style>
                        <div class="image-container">
                            <img src="{image_url}" class="image-bordure">
                        </div>
                    """, unsafe_allow_html=True)
                except IndexError:
                    st.error(f"Le film '{titre_film}' n'existe pas dans la base de données.")

        with st.sidebar.expander("Choisissez éléments TMDB dans résultats", expanded=False):
            # Curseur pour ajuster le nombre de films similaires, avec plage de 1 à 10
            n_similaire = st.slider(
                "Nombre de films similaires souhaités :", 
                min_value=1,  # Valeur minimale
                max_value=10,  # Valeur maximale
                value=5,  # Valeur par défaut
                step=1  # Incrément de 1
            )
            # Nouveau curseur pour ajuster le nombre de films à afficher par ligne
            n_par_ligne = st.slider(
                "Nombre de films à afficher par ligne :", 
                min_value=1,  # Valeur minimale
                max_value=5,  # Valeur maximale
                value=5,  # Valeur par défaut
                step=1  # Incrément de 1
            )

            # Créer deux colonnes dans l'expander pour afficher les checkboxes
            col1, col2 = st.columns(2)

            # Afficher les checkboxes dans les colonnes
            with col1:
                show_image = st.checkbox("Affiche", value=True)
                show_video = st.checkbox("Vidéo", value=True)     
                show_realisateur = st.checkbox("Réalisateur", value=True)
                show_date_sortie = st.checkbox("Sortie", value=True)
                show_duree = st.checkbox("Durée", value=True)
                show_acteurs = st.checkbox("Acteurs", value=True)
            with col2:
                show_genres = st.checkbox("Genre", value=True)
                show_note = st.checkbox("Note", value=True)
                show_synopsis = st.checkbox("Synopsis", value=True)
                show_budget = st.checkbox("Budget", value=True)
                show_revenu = st.checkbox("Revenu", value=True)

            # n_acteurs = st.slider(
            #     "Acteurs à afficher :", 
            #     min_value=1,  # Valeur minimale
            #     max_value=10,  # Valeur maximale
            #     value=2,  # Valeur par défaut
            #     step=1  # Incrément de 1
            # )

                
        # Afficher la taille actuelle en pixels
        with st.sidebar.expander("Accessibilité", expanded=False):
             # Utilisation de checkbox pour simuler un toggle
            mode_sombre = st.checkbox("Mode sombre", value=True)

            st.markdown(f"""<div style="text-align: center; font-size: {st.session_state.taille_texte}px;">
            <strong>Taille actuelle : {st.session_state.taille_texte}px</strong>
            </div>""", unsafe_allow_html=True)

            # Titre pour les boutons de changement de taille de police
            st.markdown("<h3 style='font-size: 13px; font-weight: normal;'>Changer la taille de la police</h3>", unsafe_allow_html=True)
            # Utilisation de st.columns pour centrer les boutons "Agrandir", "Réduire" et "Défaut"
            col1, col2, col3 = st.columns([1, 1, 1])  # Crée trois colonnes avec un ratio égal
            with col1:
                if st.button("Moins"):
                    st.session_state.taille_texte -= 2
            with col2:
                if st.button("Défaut"):  # Bouton pour réinitialiser la taille
                    st.session_state.taille_texte = 16
            with col3:
                if st.button("Plus"):
                    st.session_state.taille_texte += 2

            # Limiter la taille du texte pour éviter des valeurs trop petites ou trop grandes
            st.session_state.taille_texte = max(10, min(st.session_state.taille_texte, 30))  # Plage de 10 à 30

if titre_film:
    films_similaires = recommander_films(titre_film, n_similaire)
    if isinstance(films_similaires, str):
        st.error(films_similaires)
    else:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <img src="https://i.ibb.co/0RmGv6Rx/Logo-Decodex.jpg" width="150" />
                <h1 style="text-align: center; flex-grow: 1;">Système de recommandations de Films</h1>
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCqEbRPjpDdbE5LWGX26B8LOaUgTERrLam2Q&s" width="150" />
            </div>
            <div style="height: 20px;"></div>  <!-- Espace après le bloc -->
            """,
            unsafe_allow_html=True
            )

        # Mise du titre des films similaires en rouge et avec taille uniforme
        st.markdown(f"<div style='font-size:{st.session_state.taille_texte + 8}px;'><b>Les {n_similaire} Films les plus similaires à <span style='color:red;'>{titre_film}</b></span> :</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        # Calcul du nombre de lignes nécessaires en fonction du nombre de films par ligne (n_par_ligne)
        lignes = (n_similaire // n_par_ligne) + (1 if n_similaire % n_par_ligne != 0 else 0)

        # Affichage des films similaires en utilisant st.columns, avec un nombre fixe de colonnes (n_par_ligne)
        for i in range(lignes):
            # Création des colonnes pour cette ligne avec n_par_ligne colonnes fixes
            cols = st.columns(n_par_ligne)

            # Affichage des films sur cette ligne
            for j in range(n_par_ligne):
                index = i * n_par_ligne + j
                if index < n_similaire:  # On vérifie qu'il y a encore des films à afficher
                    film_titre, film_image, film_id = films_similaires[index]

                    with cols[j]:
                        # Récupérer les détails du film
                        duree, genres, synopsis, date_sortie, acteurs, video_url, directeur, note, budget, revenu = obtenir_details_tmdb(film_id)

                        st.markdown(f"<div style='font-size:{st.session_state.taille_texte + 4}px; text-align: center; font-weight: bold;'>{film_titre}</div>", unsafe_allow_html=True)

                        # Affichage conditionnel de l'image du film
                        if show_image:
                            st.markdown(
                                f"""
                                <div style="display: flex; justify-content: center; align-items: center; 
                                            width: 100%; margin: auto;">
                                    <img src="{film_image}" style="max-height: 400px; border: 1px solid black; border-radius: 8px; object-fit: cover;">
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )

                        # Ajout d'un espace de 20px après l'image
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                        # Affichage conditionnel de la vidéo YouTube
                        if show_video and video_url:
                            st.video(video_url)
                        elif show_video:
                            st.warning("Aucune vidéo disponible pour ce film.")

                        # Affichage conditionnel des autres informations
                        if show_realisateur:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Réalisateur :</u> </strong>{directeur}</div>""", unsafe_allow_html=True)
                        if show_date_sortie:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Date de sortie :</u> </strong>{date_sortie}</div>""", unsafe_allow_html=True)
                        if show_duree:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Durée :</u> </strong>{duree} minutes</div>""", unsafe_allow_html=True)
                        if show_genres:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Genres :</u> </strong>{', '.join(genres)}</div>""", unsafe_allow_html=True)
                        if show_acteurs and acteurs:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Acteurs :</u> </strong>{', '.join(acteurs)}</div>""", unsafe_allow_html=True)
                        if show_note:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Note moyenne :</u> </strong>{note:.1f}/10</div>""", unsafe_allow_html=True)
                        if show_budget and budget > 0:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Budget :</u> </strong>{budget:,.0f} $</div>""", unsafe_allow_html=True)
                        if show_revenu and revenu > 0:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Revenus :</u> </strong>{revenu:,.0f} $</div>""", unsafe_allow_html=True)    
                        if show_synopsis:
                            st.markdown(f"""<div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px;"><strong><u>Synopsis :</u> </strong></div>""", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="font-size:{st.session_state.taille_texte}px; margin-bottom: 10px; text-align: justify;">
                            {synopsis}
                            </div>
                            """, unsafe_allow_html=True)

            # Si ce n'est pas la dernière ligne, ajouter une bordure
            if i < lignes - 1:
                st.markdown("<div style='border-top: 5px solid black;'></div>", unsafe_allow_html=True)
