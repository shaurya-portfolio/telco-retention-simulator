import os
import time
import pickle
import pandas as pd
import numpy as np
from google.cloud import bigquery
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()

#state management
offset_file = 'config/pipeline_offset.txt'
try:
    with open(offset_file, 'r') as f:
        current_offset = int(f.read().strip())
except FileNotFoundError:
    current_offset = 0

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/gcp_credentials.json"
client = bigquery.Client()
target_table = "telco_dataset.live_customers"
batch_size = 50


with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/lin_reg_model.pkl', 'rb') as f:
    lin_reg = pickle.load(f)
with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)

def trigger_synthetic_generator(client, df_reference, table_id, num_rows=50):
    print(f"\n⚠️ CRITICAL: Live database empty. Generating {num_rows} synthetic customers...")
    fake_data = pd.DataFrame()
    for col in df_reference.columns:
        if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            mean = df_reference[col].mean()
            std = df_reference[col].std()
            fake_data[col] = np.random.normal(loc=mean, scale=std, size=num_rows)
            fake_data[col] = np.clip(fake_data[col], 0, None)
        else:
            probs = df_reference[col].value_counts(normalize=True)
            fake_data[col] = np.random.choice(probs.index, p=probs.values, size=num_rows)
    
    job = client.load_table_from_dataframe(fake_data, table_id)
    job.result()
    print("✅ Synthetic data injected into BigQuery!")
    return fake_data



print("--- Starting Production Data Pipeline ---")


query = f"SELECT * FROM `{target_table}` LIMIT {batch_size} OFFSET {current_offset}"
df_batch = client.query(query).to_dataframe()

if df_batch.empty:
    df_historical = pd.read_csv('data/historical_train_data.csv')
    trigger_synthetic_generator(client, df_historical, target_table)

if 'customerID' not in df_batch.columns:
    import random
    df_batch['customerID'] = [f"CUST-{random.randint(10000, 99999)}" for _ in range(len(df_batch))]
    
    
X_live = df_batch.drop(columns=['customerID','Churn', 'TotalCharges'], errors='ignore')

X_live_scaled = scaler.transform(X_live)
df_batch['Predicted_Churn_Risk'] = log_reg.predict(X_live_scaled)
df_batch['Predicted_CLV'] = lin_reg.predict(X_live_scaled)

high_value_at_risk = df_batch[(df_batch['Predicted_Churn_Risk'] == 1) & (df_batch['Predicted_CLV'] > 2000)]

print(f"[Batch {current_offset + 1}-{current_offset + batch_size}] Processed. Found {len(high_value_at_risk)} at-risk profiles.")
#actionable email alert
if not high_value_at_risk.empty:
   print("Exporting alerts for Web Dashboard...")
   alerts_file_path = 'data/latest_alerts.csv'
   high_value_at_risk.to_csv(alerts_file_path, index=False)
   print("Triggering Email alert to Retention Team...")

   msg = EmailMessage()
   msg.set_content(f"URGENT:\n\n{len(high_value_at_risk)} High-Value customers (CLV > $2000) are predicted to churn in today's batch.\n\nClick here to view full profiles and run what-if scenarios:\n👉 http://127.0.0.1:5000\n\nInitiate retention protocols immediately.")   
   msg['Subject'] = '🚨 Automated Alert: High-Value Churn Risk Detected'
   msg['From'] = 'shauryasinhaop@gmail.com'
   msg['To'] = 'shauryasinha070@gmail.com'
   try:
           server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
           app_password = os.getenv('GMAIL_APP_PASSWORD')
           server.login('shauryasinhaop@gmail.com', app_password)
           server.send_message(msg)
           server.quit()
           print("Email dispatched successfully.")
   except Exception as e:
           print(f"Failed to send email: {e}")

    
with open(offset_file, 'w') as f:
    f.write(str(current_offset + batch_size))
    
print("Pipeline Execution Completed.")