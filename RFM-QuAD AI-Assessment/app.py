# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

app = Flask(__name__)
model = pickle.load(open("kmeans_model.pkl", 'rb'))

def load_and_prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    
    # Convert CustomerID to String and create Amount column
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    
    # Compute RFM metrics
    rfm_m = df.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
    max_date = max(df['InvoiceDate'])
    df['Diff'] = max_date - df['InvoiceDate']
    rfm_p = df.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    
    # Remove outliers
    def remove_outliers(rfm, columns):
        for col in columns:
            Q1 = rfm[col].quantile(0.25)
            Q3 = rfm[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            rfm = rfm[(rfm[col] >= lower_bound) & (rfm[col] <= upper_bound)]
        return rfm

    # Remove outliers from 'Amount', 'Frequency', and 'Recency'
    df_cleaned = remove_outliers(rfm, ['Amount', 'Frequency', 'Recency'])
    
    return df_cleaned

def preprocess_data(file_path):
    df_cleaned = load_and_prepare_data(file_path)
    df_cleaned = df_cleaned[['CustomerID', 'Amount', 'Frequency', 'Recency']]

    # Instantiate scaler
    scaler = StandardScaler()

    # fit_transform
    rfm_df_scaled = scaler.fit_transform(df_cleaned[['Amount', 'Frequency', 'Recency']])
    
    return df_cleaned, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files["file"]
        file_path = os.path.join(os.getcwd(), file.filename)
        file.save(file_path)
        df, df_scaled = preprocess_data(file_path)
        result_df = model.predict(df_scaled)
        
        df['Cluster_ID'] = result_df
        
        #result_csv_path = os.path.json(os.getcwd(), 'customer_segment.csv')
        #df.to_csv(result_csv_path, index = False)
        #df[['Amount', 'Frequency', 'Recency', 'Cluster']]

        # Generate the images and save
        sns.stripplot(x='Cluster_ID', y='Amount', data=df, hue='Cluster_ID')
        amount_img_path = 'static/ClusterID_Amount.png'
        plt.savefig(amount_img_path)
        plt.clf()

        sns.stripplot(x='Cluster_ID', y='Frequency', data=df, hue='Cluster_ID')
        freq_img_path = 'static/ClusterID_Frequency.png'
        plt.savefig(freq_img_path)
        plt.clf()

        sns.stripplot(x='Cluster_ID', y='Recency', data=df, hue='Cluster_ID')
        recency_img_path = 'static/ClusterID_Recency.png'
        plt.savefig(recency_img_path)
        plt.clf()
        
        # Convert DataFrame to JSON
        df_json = df.to_json(orient='records')
        
        # Return the filenames of the generated images as a JSON response
        response = {
            'amount_img': amount_img_path,
            'freq_img': freq_img_path,
            'recency_img': recency_img_path
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

# For cloud
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8080)
