from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model, encoder, and scaler
model = joblib.load('models/random_forest_model.pkl')
encoder = joblib.load('models/encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract data from the request
    item_name = data['Item Name']
    brand = data['Brand']
    item_price = data['Item Price']
    
    # Prepare data for prediction
    input_data = pd.DataFrame([[item_name, brand, item_price]], columns=['Item Name', 'Brand', 'Item Price'])
    input_encoded = encoder.transform(input_data[['Item Name', 'Brand']])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Item Name', 'Brand']))
    input_prepared = pd.concat([input_encoded_df, input_data[['Item Price']].reset_index(drop=True)], axis=1)
    input_prepared[['Item Price']] = scaler.transform(input_prepared[['Item Price']])
    
    # Make prediction
    prediction = model.predict(input_prepared)
    
    # Send response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
