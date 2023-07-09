from flask import Blueprint, render_template, send_from_directory, request
import json
import joblib

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    with open('TDR\datos.json', 'r') as file:
        acc_gb_par = json.load(file)
        mean_accuracy_gb_par = acc_gb_par['mean_accuracy_gb_par']

    
    model_lg = joblib.load('TDR\models\model_lg.pkl')
    model_dt = joblib.load('TDR\models\model_dt.pkl')
    model_knn = joblib.load('TDR\models\model_knn.pkl')
    model_svm = joblib.load('TDR\models\model_svm.pkl')
    model_nb = joblib.load('TDR\models\model_nb.pkl')
    model_gb = joblib.load('TDR\models\model_gb.pkl')
    model_rf = joblib.load('TDR\models\model_rf.pkl')
    model_ann = joblib.load('TDR\models\model_ann.pkl')
    model_lg_diabetes = joblib.load('TDR\models\model_lg_diabetes.pkl')
    model_dt_diabetes = joblib.load('TDR\models\model_dt_diabetes.pkl')
    model_knn_diabetes = joblib.load('TDR\models\model_knn_diabetes.pkl')
    model_svm_diabetes = joblib.load('TDR\models\model_svm_diabetes.pkl')
    model_nb_diabetes = joblib.load('TDR\models\model_nb_diabetes.pkl')
    model_gb_diabetes = joblib.load('TDR\models\model_gb_diabetes.pkl')
    model_rf_diabetes = joblib.load('TDR\models\model_rf_diabetes.pkl')
    model_ann_db = joblib.load('TDR\models\model_ann_db.pkl')
    model_lg_hep = joblib.load('TDR\models\model_lg_hep.pkl')
    model_dt_hep = joblib.load('TDR\models\model_dt_hep.pkl')
    model_knn_hep = joblib.load('TDR\models\model_knn_hep.pkl')
    model_svm_hep = joblib.load('TDR\models\model_svm_hep.pkl')
    model_nb_hep = joblib.load('TDR\models\model_nb_hep.pkl')
    model_gb_hep = joblib.load('TDR\models\model_gb_hep.pkl')
    model_rf_hep = joblib.load('TDR\models\model_rf_hep.pkl')
    model_ann_hep = joblib.load('TDR\models\model_ann_hep.pkl')
    model_lg_par = joblib.load('TDR\models\model_lg_par.pkl')
    model_dt_par = joblib.load('TDR\models\model_dt_par.pkl')
    model_knn_par = joblib.load('TDR\models\model_knn_par.pkl')
    model_svm_par = joblib.load('TDR\models\model_svm_par.pkl')
    model_nb_par = joblib.load('TDR\models\model_nb_par.pkl')
    model_gb_par = joblib.load('TDR\models\model_gb_par.pkl')
    model_rf_par = joblib.load('TDR\models\model_rf_par.pkl')
    model_ann_par = joblib.load('TDR\models\model_ann_par.pkl')
    model_lg_hcc = joblib.load('TDR\models\model_lg_hcc.pkl')
    model_dt_hcc = joblib.load('TDR\models\model_dt_hcc.pkl')
    model_knn_hcc = joblib.load('TDR\models\model_knn_hcc.pkl')
    model_svm_hcc = joblib.load('TDR\models\model_svm_hcc.pkl')
    model_nb_hcc = joblib.load('TDR\models\model_nb_hcc.pkl')
    model_gb_hcc = joblib.load('TDR\models\model_gb_hcc.pkl')
    model_rf_hcc = joblib.load('TDR\models\model_rf_hcc.pkl')
    model_ann_hcc = joblib.load('TDR\models\model_ann_hcc.pkl')
    models = {
    'model_lg': model_lg,
    'model_dt': model_dt,
    'model_knn': model_knn,
    'model_svm': model_svm,
    'model_nb': model_nb,
    'model_gb': model_gb,
    'model_rf': model_rf,
    'model_ann': model_ann,
    'model_lg_diabetes': model_lg_diabetes,
    'model_dt_diabetes': model_dt_diabetes,
    'model_knn_diabetes': model_knn_diabetes,
    'model_svm_diabetes': model_svm_diabetes,
    'model_nb_diabetes': model_nb_diabetes,
    'model_gb_diabetes': model_gb_diabetes,
    'model_rf_diabetes': model_rf_diabetes,
    'model_ann_db': model_ann_db,
    'model_lg_hep': model_lg_hep,
    'model_dt_hep': model_dt_hep,
    'model_knn_hep': model_knn_hep,
    'model_svm_hep': model_svm_hep,
    'model_nb_hep': model_nb_hep,
    'model_gb_hep': model_gb_hep,
    'model_rf_hep': model_rf_hep,
    'model_ann_hep': model_ann_hep,
    'model_lg_par': model_lg_par,
    'model_dt_par': model_dt_par,
    'model_knn_par': model_knn_par,
    'model_svm_par': model_svm_par,
    'model_nb_par': model_nb_par,
    'model_gb_par': model_gb_par,
    'model_rf_par': model_rf_par,
    'model_ann_par': model_ann_par,
    'model_lg_hcc': model_lg_hcc,
    'model_dt_hcc': model_dt_hcc,
    'model_knn_hcc': model_knn_hcc,
    'model_svm_hcc': model_svm_hcc,
    'model_nb_hcc': model_nb_hcc,
    'model_gb_hcc': model_gb_hcc,
    'model_rf_hcc': model_rf_hcc,
    'model_ann_hcc': model_ann_hcc
    }
    


    # Aquí puedes procesar la predicción utilizando el algoritmo y el dataset
    # Código para cargar el dataset, aplicar el algoritmo y generar la predicción

    # Devolver el resultado de la predicción
    return render_template("index.html", models=models)
@views.route('/predict', methods=['POST'])
def guardar_datos():
    enfermedad = request.form['enfermedad']
    algoritmo = request.form['algoritmo']

    model_lg = joblib.load('TDR\models\model_lg.pkl')
    model_dt = joblib.load('TDR\models\model_dt.pkl')
    model_knn = joblib.load('TDR\models\model_knn.pkl')
    model_svm = joblib.load('TDR\models\model_svm.pkl')
    model_nb = joblib.load('TDR\models\model_nb.pkl')
    model_gb = joblib.load('TDR\models\model_gb.pkl')
    model_rf = joblib.load('TDR\models\model_rf.pkl')
    model_ann = joblib.load('TDR\models\model_ann.pkl')
    model_lg_diabetes = joblib.load('TDR\models\model_lg_diabetes.pkl')
    model_dt_diabetes = joblib.load('TDR\models\model_dt_diabetes.pkl')
    model_knn_diabetes = joblib.load('TDR\models\model_knn_diabetes.pkl')
    model_svm_diabetes = joblib.load('TDR\models\model_svm_diabetes.pkl')
    model_nb_diabetes = joblib.load('TDR\models\model_nb_diabetes.pkl')
    model_gb_diabetes = joblib.load('TDR\models\model_gb_diabetes.pkl')
    model_rf_diabetes = joblib.load('TDR\models\model_rf_diabetes.pkl')
    model_ann_db = joblib.load('TDR\models\model_ann_db.pkl')
    model_lg_hep = joblib.load('TDR\models\model_lg_hep.pkl')
    model_dt_hep = joblib.load('TDR\models\model_dt_hep.pkl')
    model_knn_hep = joblib.load('TDR\models\model_knn_hep.pkl')
    model_svm_hep = joblib.load('TDR\models\model_svm_hep.pkl')
    model_nb_hep = joblib.load('TDR\models\model_nb_hep.pkl')
    model_gb_hep = joblib.load('TDR\models\model_gb_hep.pkl')
    model_rf_hep = joblib.load('TDR\models\model_rf_hep.pkl')
    model_ann_hep = joblib.load('TDR\models\model_ann_hep.pkl')
    model_lg_par = joblib.load('TDR\models\model_lg_par.pkl')
    model_dt_par = joblib.load('TDR\models\model_dt_par.pkl')
    model_knn_par = joblib.load('TDR\models\model_knn_par.pkl')
    model_svm_par = joblib.load('TDR\models\model_svm_par.pkl')
    model_nb_par = joblib.load('TDR\models\model_nb_par.pkl')
    model_gb_par = joblib.load('TDR\models\model_gb_par.pkl')
    model_rf_par = joblib.load('TDR\models\model_rf_par.pkl')
    model_ann_par = joblib.load('TDR\models\model_ann_par.pkl')
    model_lg_hcc = joblib.load('TDR\models\model_lg_hcc.pkl')
    model_dt_hcc = joblib.load('TDR\models\model_dt_hcc.pkl')
    model_knn_hcc = joblib.load('TDR\models\model_knn_hcc.pkl')
    model_svm_hcc = joblib.load('TDR\models\model_svm_hcc.pkl')
    model_nb_hcc = joblib.load('TDR\models\model_nb_hcc.pkl')
    model_gb_hcc = joblib.load('TDR\models\model_gb_hcc.pkl')
    model_rf_hcc = joblib.load('TDR\models\model_rf_hcc.pkl')
    model_ann_hcc = joblib.load('TDR\models\model_ann_hcc.pkl')
    models = {
    'model_lg': model_lg,
    'model_dt': model_dt,
    'model_knn': model_knn,
    'model_svm': model_svm,
    'model_nb': model_nb,
    'model_gb': model_gb,
    'model_rf': model_rf,
    'model_ann': model_ann,
    'model_lg_diabetes': model_lg_diabetes,
    'model_dt_diabetes': model_dt_diabetes,
    'model_knn_diabetes': model_knn_diabetes,
    'model_svm_diabetes': model_svm_diabetes,
    'model_nb_diabetes': model_nb_diabetes,
    'model_gb_diabetes': model_gb_diabetes,
    'model_rf_diabetes': model_rf_diabetes,
    'model_ann_db': model_ann_db,
    'model_lg_hep': model_lg_hep,
    'model_dt_hep': model_dt_hep,
    'model_knn_hep': model_knn_hep,
    'model_svm_hep': model_svm_hep,
    'model_nb_hep': model_nb_hep,
    'model_gb_hep': model_gb_hep,
    'model_rf_hep': model_rf_hep,
    'model_ann_hep': model_ann_hep,
    'model_lg_par': model_lg_par,
    'model_dt_par': model_dt_par,
    'model_knn_par': model_knn_par,
    'model_svm_par': model_svm_par,
    'model_nb_par': model_nb_par,
    'model_gb_par': model_gb_par,
    'model_rf_par': model_rf_par,
    'model_ann_par': model_ann_par,
    'model_lg_hcc': model_lg_hcc,
    'model_dt_hcc': model_dt_hcc,
    'model_knn_hcc': model_knn_hcc,
    'model_svm_hcc': model_svm_hcc,
    'model_nb_hcc': model_nb_hcc,
    'model_gb_hcc': model_gb_hcc,
    'model_rf_hcc': model_rf_hcc,
    'model_ann_hcc': model_ann_hcc
    }

    # Aquí es donde realizarías la predicción usando el modelo correspondiente
    if algoritmo == 'algoritmo1' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_lg']
    elif algoritmo == 'algoritmo2' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_dt']
    elif algoritmo == 'algoritmo3' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_knn']
    elif algoritmo == 'algoritmo4' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_svm']
    elif algoritmo == 'algoritmo5' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_nb']
    elif algoritmo == 'algoritmo6' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_gb']
    elif algoritmo == 'algoritmo7' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_rf']
    elif algoritmo == 'algoritmo8' and enfermedad == 'breast_cancer':
        modelo_seleccionado = models['model_ann']
    elif algoritmo == 'algoritmo1' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_lg_diabetes']
    elif algoritmo == 'algoritmo2' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_dt_diabetes']
    elif algoritmo == 'algoritmo3' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_knn_diabetes']
    elif algoritmo == 'algoritmo4' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_svm_diabetes']
    elif algoritmo == 'algoritmo5' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_nb_diabetes']
    elif algoritmo == 'algoritmo6' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_gb_diabetes']
    elif algoritmo == 'algoritmo7' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_rf_diabetes']
    elif algoritmo == 'algoritmo8' and enfermedad == 'diabetes':
        modelo_seleccionado = models['model_ann_db']
    elif algoritmo == 'algoritmo1' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_lg_hep']
    elif algoritmo == 'algoritmo2' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_dt_hep']
    elif algoritmo == 'algoritmo3' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_knn_hep']
    elif algoritmo == 'algoritmo4' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_svm_hep']
    elif algoritmo == 'algoritmo5' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_nb_hep']
    elif algoritmo == 'algoritmo6' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_gb_hep']
    elif algoritmo == 'algoritmo7' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_rf_hep']
    elif algoritmo == 'algoritmo8' and enfermedad == 'hepatitis':
        modelo_seleccionado = models['model_ann_hep']
    elif algoritmo == 'algoritmo1' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_lg_par']
    elif algoritmo == 'algoritmo2' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_dt_par']
    elif algoritmo == 'algoritmo3' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_knn_par']
    elif algoritmo == 'algoritmo4' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_svm_par']
    elif algoritmo == 'algoritmo5' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_nb_par']
    elif algoritmo == 'algoritmo6' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_gb_par']
    elif algoritmo == 'algoritmo7' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_rf_par']
    elif algoritmo == 'algoritmo8' and enfermedad == 'parkinson':
        modelo_seleccionado = models['model_ann_par']
    elif algoritmo == 'algoritmo1' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_lg_hcc']
    elif algoritmo == 'algoritmo2' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_dt_hcc']
    elif algoritmo == 'algoritmo3' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_knn_hcc']
    elif algoritmo == 'algoritmo4' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_svm_hcc']
    elif algoritmo == 'algoritmo5' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_nb_hcc']
    elif algoritmo == 'algoritmo6' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_gb__hcc']
    elif algoritmo == 'algoritmo7' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_rf_hcc']
    elif algoritmo == 'algoritmo8' and enfermedad == 'HCC':
        modelo_seleccionado = models['model_ann_hcc']
    else:
        modelo_seleccionado = None

    
    data = request.form.to_dict()  
    valores = []
    for clave, valor in data.items():
        if clave.startswith('valor') and valor != '':
            valor_numerico = float(valor)
            valores.append(valor_numerico)
    try:
        resultado = modelo_seleccionado.predict([valores])
        if resultado == [1]:
            resultado = 'positive diagnostic, please consider consulting a doctor'
        elif resultado == [0]:
            resultado = "negative diagnostic, all good!"
        else:
            resultado = None
        mensaje_personalizado = None
    except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"
        resultado = None
        
    

    return render_template('resultado.html', resultado=resultado, error=mensaje_personalizado)

