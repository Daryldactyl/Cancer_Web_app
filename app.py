import streamlit as st
from main import import_data, train_ksvm, train_rfc, train_xgboost, train_linear, get_radar_chart, make_pred

st.set_page_config(page_title='Breast Cancer Predictor',
                       layout='wide',
                       initial_sidebar_state='expanded')

if __name__ == '__main__':
    # Set up the classes
    classes = {0: 'Benign', 1: 'Malignant'}

    # Build the dataset
    (x_train, x_test, y_train, y_test, sc,
     columns, pretty_columns, max_values, mean_values) = import_data()

    # Build all models
    xg_model, xg_cr, xg_acc = train_xgboost(x_train, y_train, x_test, y_test)

    lin_model, lin_cr, lin_acc = train_linear(x_train, y_train, x_test, y_test)

    rfc_model, rf_cr, rf_acc = train_rfc(x_train, y_train, x_test, y_test)

    # Best Model:
    ksvm_model, ksvm_cr, ksvm_acc = train_ksvm(x_train, y_train, x_test, y_test)

    # Streamlit app
    st.markdown('# Breast Cancer Predictor')
    st.write("""
    Diagnose Breast Cancer from Tissue Sample.
     Input Measurements Gathered From Your Tissue Sample Into the Sidebar""")

    st.sidebar.header('Cell Tissue Details')
    slider_labels = pretty_columns

    input_dict = {}

    # Create Sliders in Sidebar
    for (label, column,
         max_value, mean_value) in zip(slider_labels, columns,
                                       max_values, mean_values):
        input_dict[column] = st.sidebar.slider(label=label,
                          min_value=0.0,
                          max_value=float(max_value),
                          value=mean_value)

    col1, col2 = st.columns([4,1])

    with col1:
        selected_model = st.selectbox('Select Model:', ['XGBoost', 'Linear', 'Random Forest', 'Best: Kernel SVM'], index=3)
        radar = get_radar_chart(input_data=input_dict)
        st.plotly_chart(radar)

    with col2:
        if selected_model == 'XGBoost':
            model = xg_model
            acc = xg_acc
            cr = xg_cr
        elif selected_model == 'Linear':
            model = lin_model
            acc = lin_acc
            cr = lin_cr
        elif selected_model == 'Random Forest':
            model = rfc_model
            acc = rf_acc
            cr = rf_cr
        else:
            model = ksvm_model
            acc = ksvm_acc
            cr = ksvm_cr
        make_pred(input_dict, sc, model, acc, cr)

