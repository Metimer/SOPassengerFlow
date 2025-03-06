import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


# Configuration de la page
st.set_page_config(
    page_title="San Francisco Airport Passenger Flow",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour améliorer le design
st.markdown(
    """
    <style>
    /* Fond général de l'application */
    .stApp {
        background-color: #1D1E33 !important;
        color: white !important;
    }
    label {
    color: white !important;
}

    /* Style de la sidebar */
    [data-testid="stSidebar"] {
        background-color: black !important;
        color: black !important;
    }
    
    /* Modifier la couleur des labels */
    [data-testid="stSidebar"] label {
        color: white !important; /* Texte des labels en blanc */
    }
    
    /* Modifier le style des boutons */
    div.stButton > button {
    color: black !important;  /* Texte en noir */
    background-color: #FFD700 !important;  /* Fond doré */
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    transition: 0.3s;
    }

    div.stButton > button:hover {
    background-color: #E5C100 !important; /* Couleur légèrement plus foncée au survol */
    }
    
    /* Style pour le DataFrame */
    [data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 2px solid #FFD700;
    background-color: black;
    color: white;
    font-size: 18px !important;
    font-weight: bold;
    color: white !important;
    }

    [data-testid="stTable"] tbody tr td {
    font-size: 18px;
    text-align: center;
     }

    /* Augmenter la taille du texte dans les colonnes */
    div[data-testid="stDataEditor"] td {
            font-size: 18px !important;
            font-weight: bold;
    }

    /* Style des titres */
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #FFD700;
        font-family: "Georgia", serif; /* Police élégante et disponible par défaut */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        letter-spacing: 2px; /* Espacement des lettres pour un effet premium */
        text-transform: uppercase; /* Mise en majuscules pour plus d'impact */
        -webkit-text-stroke: 1px black; /* Contour fin pour améliorer la lisibilité */
    }

    /* Style des cartes */
    .film-card {
        background-color: black ;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    /* Style des affiches */
    .film-card img {
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Style du texte des cartes */
    .film-card h3 {
        text-align: center;
        color: #FFD700;
        font-family: "Georgia", serif; /* Police élégante et disponible par défaut */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        letter-spacing: 2px; /* Espacement des lettres pour un effet premium */       
        -webkit-text-stroke: 1px black; /* Contour fin pour améliorer la lisibilité */
    }

    .film-card p {
        text-align: center;
        font-size: 17px;
        color: #CCCCCC;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        font-family: "Georgia", serif; /* Police élégante et disponible par défaut */
    }

    .film-card h4 {
    font-size: 14px;
    color: #FFD700;
    font-family: "Georgia", serif;
    height: 50px;  
    text-align: center;
    display: flex;
    align-items: center; 
    justify-content: center;
}

    """,
    unsafe_allow_html=True
)

# Logo dans la sidebar
logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/SFO_San_Francisco_International_Airport_Logo.svg/2560px-SFO_San_Francisco_International_Airport_Logo.svg.png"
st.sidebar.image(logo_url, use_container_width=True)

# Chargement des données
loading_gif = "https://i.gifer.com/MCn2.gif"  
gif_container = st.empty()

# Afficher le GIF dans le conteneur
gif_container.image(loading_gif, use_container_width=True)

@st.cache_data
def load_data():
    df = pd.read_csv('https://drive.google.com/uc?id=14b1ajldtzycooI5h5WAKjbpB_CzVfIlv')
    df.drop(['operating_airline','operating_airline_iata_code','published_airline','published_airline_iata_code',
             'geo_region','price_category_code','data_as_of','data_loaded_at','activity_period'], axis=1, inplace=True)
    
    df['activity_period_start_date'] = pd.to_datetime(df['activity_period_start_date'])
    df['année'] = df['activity_period_start_date'].dt.year
    df['mois'] = df['activity_period_start_date'].dt.month
    df = df.drop('activity_period_start_date', axis=1).groupby(
        by=['année', 'mois','terminal','boarding_area','geo_summary','activity_type_code'], as_index=False).sum()
    
    df = pd.get_dummies(df, columns=['geo_summary','activity_type_code','terminal','boarding_area'], dtype=int)
    return df

@st.cache_resource
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             random_state=42,
                             subsample=0.8,
                             n_estimators=500,
                             max_depth=9,
                             min_child_weight=1,
                             learning_rate=0.05,
                             gamma=0.3,
                             colsample_bytree=0.9)
    model.fit(X_train, y_train)
    return model

# Charger les données
dfml = load_data()
X = dfml.drop('passenger_count', axis=1).values  
y = dfml['passenger_count']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Entraîner le modèle avec mise en cache
xgb_model = train_model(X_train, y_train)



gif_container.empty()  # Efface le GIF


# Menu

menu_options = ["Passenger Flow Estimator", "About us"]
selection = st.sidebar.selectbox("Choose a section", menu_options)

# Accueil

if selection == "Passenger Flow Estimator":
    st.markdown('<div class="title"> Passenger Flow Estimator </div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="film-card">
            <img src="{logo_url}" width="700">
            <p>San Francisco Airport Passenger Flow Estimator</p>
            <p>Please enter the month and year for which you want to predict passenger flow</p>             
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sélection des paramètres d'entrée (année, mois, terminal)
    annee_input = st.number_input("Choose Year (ex: 2025)", min_value=2000, max_value=2030, value=2025, step=1)
    mois_input = st.selectbox("Choose Month", list(range(1, 13)), index=2)
    
    # Déclarer la liste des terminaux
    terminals = ["International", "Other", "Terminal 1", "Terminal 2", "Terminal 3"]
    terminal_input = st.selectbox("Choose Terminal", terminals)

    if st.button("Predict"):
        # Création du DataFrame d'entrée pour toutes les combinaisons possibles
        geo_summary_options = ["Domestic", "International"]
        activity_types = ["Deplaned", "Enplaned", "Thru / Transit"]
        boarding_areas = ["A", "B", "C", "D", "E", "F", "G", "Other"]

        # Créer un DataFrame avec toutes les combinaisons possibles
        df_input = pd.DataFrame(
            [(annee_input, mois_input, geo_summary, activity_type, terminal_input, boarding_area)
             for geo_summary in geo_summary_options
             for activity_type in activity_types
             for boarding_area in boarding_areas],
            columns=["année", "mois", "geo_summary", "activity_type_code", "terminal", "boarding_area"]
        )

        # Transformation en variables dummies
        df_input = pd.get_dummies(df_input, columns=['geo_summary', 'activity_type_code', 'terminal', 'boarding_area'], dtype=int)

        # Ajouter les colonnes manquantes
        features_model = ["année", "mois", "geo_summary_Domestic", "geo_summary_International",
                          "activity_type_code_Deplaned", "activity_type_code_Enplaned", "activity_type_code_Thru / Transit",
                          "terminal_International", "terminal_Other", "terminal_Terminal 1",
                          "terminal_Terminal 2", "terminal_Terminal 3",
                          "boarding_area_A", "boarding_area_B", "boarding_area_C", "boarding_area_D",
                          "boarding_area_E", "boarding_area_F", "boarding_area_G", "boarding_area_Other"]

        for col in features_model:
            if col not in df_input.columns:
                df_input[col] = 0  # Ajouter les colonnes manquantes avec la valeur 0

        # Réorganiser les colonnes pour correspondre aux colonnes du modèle
        df_input = df_input[features_model]

        # Prédiction
        df_input["Predicted Passenger Count"] = xgb_model.predict(df_input).round(0).astype(int)
        df_input["Predicted Passenger Count"] = df_input["Predicted Passenger Count"].clip(0, None).astype(int)
        

        # Récupération des valeurs de geo_summary et activity_type_code
        df_input["geo_summary"] = df_input[['geo_summary_Domestic', 'geo_summary_International']].idxmax(axis=1).str.replace('geo_summary_', '')
        df_input["activity_type_code"] = df_input[['activity_type_code_Deplaned', 'activity_type_code_Enplaned', 'activity_type_code_Thru / Transit']].idxmax(axis=1).str.replace('activity_type_code_', '')

        # Récupération des valeurs de boarding_area
        df_input["boarding_area"] = df_input[['boarding_area_A', 'boarding_area_B', 'boarding_area_C', 'boarding_area_D',
                                            'boarding_area_E', 'boarding_area_F', 'boarding_area_G', 'boarding_area_Other']].idxmax(axis=1).str.replace('boarding_area_', '')

        # Sélection des colonnes à afficher (sans terminal)
        df_display = df_input[["année", "mois", "geo_summary", "activity_type_code", "boarding_area", "Predicted Passenger Count"]]

        # Affichage du DataFrame
        st.success("Prediction completed! Here are the estimated values:")
        st.data_editor(df_display, 
                    hide_index=True,  
                    column_config={
                        "année": st.column_config.NumberColumn("Year", format="%d"),
                        "mois": st.column_config.NumberColumn("Month", format="%d"),
                        "Predicted Passenger Count": st.column_config.NumberColumn("Passengers", format="%d")
                    })




    
# About us
elif selection == "About us":
    st.markdown('<div class="title"> Nos Partenaires </div>', unsafe_allow_html=True)
    part_url = "https://www.c5i.ai/wp-content/uploads/Analytics-Team1.png"
    part_url2='https://www.hautsdefrance-id.fr/wp-content/uploads/2022/04/wild-code-school-logo-1024x614-1.jpg'
    
    
    st.markdown(
                f"""
                <div class="film-card">
                    <h3> Noumer Data Team </h3>
                    <a href="{'https://github.com/Metimer'}" target="_blank">
                    <img src="{part_url}" width="550"></a>
                    <p>Nous sommes une équipe </p>  
                    <p>à taille humaine qui saura</p> 
                    <p>relever tout les challenges digitaux </p> 
                    <p>Pour votre entreprise !!</p> 
                    <p>N'hésitez plus , contactez nous !</p>                  
                </div>
                """,
                unsafe_allow_html=True
            )
        
    st.markdown(
                f"""                
                <div class="film-card">   
                    <h3> Wild Code School </h3> 
                    <a href="{'https://www.wildcodeschool.com/'}" target="_blank">                
                    <img src="{part_url2}" width="850"></a>
                    <p>Depuis plus de 10 ans, la Wild Code School forme des talents aux métiers de la tech et de l'IA.</p>
                    <p>Avec plus de 8 000 alumni, des formations adaptées au marché, et une pédagogie innovante,</p>
                    <p>nous préparons les professionnels de demain.</p>
                    <p>Découvrez nos spécialités pour réussir : développement web, data et IA,</p>
                    <p>IT et cybersécurité, design et gestion de projet.</p>
                    <p>Vous aurez les codes ! </p>                   
                    </div>
                """,
        unsafe_allow_html=True
            )