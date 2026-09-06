from flask import Flask,render_template,request,jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

print("Loading Models for Web Server....")
with open('models/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('models/lin_reg_model.pkl', 'rb') as f:
    lin_reg = pickle.load(f)
with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/latest_alerts',methods=['GET'])
def get_alerts():
    alerts_file = 'data/latest_alerts.csv'
    if os.path.exists(alerts_file):
        df = pd.read_csv(alerts_file)
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({'message':"No Active Alerts."})

@app.route('/api/simulate',methods=['POST'])
def simulate():
    try:
        
        customer_data = request.json
        df_input = pd.DataFrame([customer_data])
        
        
        df_input = df_input.drop(columns=['customerID', 'Churn', 'TotalCharges', 'Predicted_Churn_Risk', 'Predicted_CLV'], errors='ignore')
        
        
        expected_cols = scaler.feature_names_in_
        
        
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0
                
        
        df_input = df_input[expected_cols]
        
        
        input_scaled = scaler.transform(df_input)
        new_churn_risk = log_reg.predict(input_scaled)[0]
        new_clv = lin_reg.predict(input_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'new_churn_risk': int(new_churn_risk),
            'new_clv': round(float(new_clv), 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)





