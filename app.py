import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
    # Penguin Prediction App
    
    This app predicts the  **Palmer Penguin** species!
    
""")

st.sidebar.header('User Input Parameters')


# upload file for user input into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox(
            'Island', ('Biscoe', 'Dream', 'Torgerson'))
        sex = st.sidebar.selectbox('sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider(
            'Bill Length (mm)', 32.2, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'FLipper Length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

    # Combine user input with penguines dataset

    penguins_raw = pd.read_csv('penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df, penguins], axis=0)

    # Ordinal Encoding
    encode = ['sex', 'island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # select only first row

    # Display user iput feature
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploadded...')
        st.write(df)

    # Load Classification Model
    load_clf = pickle.load(open('penguins-clf-model.pkl', 'rb'))

    # apply model to make predictions
    prediction = load_clf.predict(df)
    predictions_proba = load_clf.predict_proba(df)

    st.subheader('Prediction')
    penguins_species = np.array(['Adele', 'Chinstrap', 'Gentoo'])
    st.write(penguins_species[prediction])

    st.subheader('Prediction Probability')
    st.write(predictions_proba)
