import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st

@st.cache_data
def clean_data():
    df = pd.read_csv('breast_cancer_tissue.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)

    df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

    return df
@st.cache_data
def import_data():
    df = pd.read_csv('breast_cancer_tissue.csv')

    max_values = df.iloc[:, 2:-1].max()
    mean_values = df.iloc[:, 2:-1].mean()

    columns = df.columns[2:].tolist()
    pretty_columns = [transform_columns(col) for col in columns]


    X = df.iloc[:, 2:-1].values
    y = df.iloc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test, sc, columns, pretty_columns, max_values, mean_values

def transform_columns(name):
    return re.sub(r'(_)', ' ', name).title()

@st.cache_resource
def train_xgboost(_x_train, _y_train, _x_test, _y_test):
    xg_model = XGBClassifier()
    xg_model.fit(_x_train, _y_train)

    #Get model accuracy
    y_pred = xg_model.predict(_x_test)

    #Evaluate
    cr = classification_report(_y_test, y_pred, output_dict=True)
    acc = accuracy_score(_y_test, y_pred)
    print(f'Accuracy of XGBoost Model: {round(acc *100, 2)}%')
    print(cr)

    return xg_model, cr, acc

@st.cache_resource
def train_linear(_x_train, _y_train, _x_test, _y_test):
    lin_model = LogisticRegression(random_state=42)
    lin_model.fit(_x_train, _y_train)

    #Get model accuracy
    y_pred = lin_model.predict(_x_test)

    #Evaluate
    cr = classification_report(_y_test, y_pred, output_dict=True)
    acc = accuracy_score(_y_test, y_pred)
    print(f'Accuracy of Linear Model: {round(acc *100, 2)}%')
    print(cr)

    return lin_model, cr, acc

@st.cache_resource
def train_rfc(_x_train, _y_train, _x_test, _y_test):
    rfc_model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    rfc_model.fit(_x_train, _y_train)

    #Get model accuracy
    y_pred = rfc_model.predict(_x_test)

    #Evaluate
    cr = classification_report(_y_test, y_pred, output_dict=True)
    acc = accuracy_score(_y_test, y_pred)
    print(f'Accuracy of Random Forest Model: {round(acc *100, 2)}%')
    print(cr)

    return rfc_model, cr, acc

@st.cache_resource
def train_ksvm(_x_train, _y_train, _x_test, _y_test):
    ksvm_model = SVC(kernel='rbf', probability=True, random_state=42)
    ksvm_model.fit(_x_train, _y_train)

    #Get model accuracy
    y_pred = ksvm_model.predict(_x_test)

    #Evaluate
    cr = classification_report(_y_test, y_pred, output_dict=True)
    acc = accuracy_score(_y_test, y_pred)
    print(f'Accuracy of Kernel SVM Model: {round(acc *100, 2)}%')
    print(cr)

    return ksvm_model, cr, acc

def make_pred(input_data, sc, model, acc, cr):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_array = sc.transform(input_array)

    prediction = model.predict(scaled_array)
    st.markdown('<p style="color: white; text-align: center; font-size: 34px;">Cell cluster prediction</p>', unsafe_allow_html=True)

    st.write(f'Model Accuracy: {round(acc * 100, 2)}%')
    with st.expander(label='Classification Report'):
        st.write(f'Precision for Positive class: {cr["1"]["precision"]}')
        st.write(f'Recall for Positive class: {cr["1"]["recall"]}')
        st.write(f'F-1 Score for Positive class: {cr["1"]["f1-score"]}')
    st.write('The cell cluster is: ')
    if prediction[0] == 0:
        st.markdown('<p style="color: green; text-align: center; font-size: 34px;">Benign</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: red; text-align: center; font-size: 34px;">Malicious</p>', unsafe_allow_html=True)
    st.write(f'Probability of benign: {round((model.predict_proba(scaled_array)[0][0] * 100), 2)}% ')
    st.write(f'Probability of malicious: {round(model.predict_proba(scaled_array)[0][1] * 100, 2)}% ')
    st.write('This model is meant to assist in diagnosis decision not as a subsitute for a professional diagnosis')


def scale_radar(input_data):
    df = clean_data()

    X = df.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/(max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict
def get_radar_chart(input_data):
    input_data = scale_radar(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                  'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
           input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
           input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
           input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


