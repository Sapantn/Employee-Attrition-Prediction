from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from db.mongo_client import mongo_client, save_prediction, get_predictions

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Global variables for model and preprocessing
model_data = None
model = None
label_encoders = {}
scaler = None
feature_columns = []
categorical_columns = []
numerical_columns = []

def load_model():
    """Load the trained model and preprocessing objects"""
    global model_data, model, label_encoders, scaler, feature_columns, categorical_columns, numerical_columns
    
    try:
        # Try to load the latest XGBoost model first
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'xgb_model.pkl')
        
        if not os.path.exists(model_path):
            # Fallback to older model
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'attrition_model.pkl')
            
        if not os.path.exists(model_path):
            print("Model file not found. Please run the training script first.")
            return False
        
        model_data = joblib.load(model_path)
        
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        categorical_columns = model_data['categorical_columns']
        numerical_columns = model_data['numerical_columns']
        
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(employee_data):
    """
    Preprocess input data for prediction
    
    Args:
        employee_data (dict): Employee information from form
    
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    try:
        print(f"🔍 Preprocessing input data: {list(employee_data.keys())}")
        
        # Create DataFrame from input data
        df = pd.DataFrame([employee_data])
        
        # Get the actual feature names from the model
        model_feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_columns
        print(f"🎯 Model expects features: {model_feature_names}")
        
        # Ensure all required features are present and in correct order
        for col in model_feature_names:
            if col not in df.columns:
                print(f"⚠️ Missing feature: {col}, adding default value 0")
                df[col] = 0  # Default value for missing features
        
        # Reorder columns to match model expectations
        df = df[model_feature_names]
        print(f"✅ DataFrame shape after reordering: {df.shape}")
        
        # Encode categorical variables (only once)
        print(f"🔤 Encoding categorical variables: {categorical_columns}")
        for col in categorical_columns:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                unique_values = le.classes_
                print(f"   Encoding {col}: {df[col].iloc[0]} -> {unique_values}")
                df[col] = df[col].apply(lambda x: x if x in unique_values else unique_values[0])
                df[col] = le.transform(df[col])
        
        # Scale numerical variables (only once)
        if numerical_columns:
            print(f"📏 Scaling numerical variables: {numerical_columns}")
            df[numerical_columns] = scaler.transform(df[numerical_columns])
        
        print(f"✅ Preprocessing completed successfully. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error preprocessing input: {e}")
        import traceback
        traceback.print_exc()
        return None

def make_prediction(employee_data):
    """
    Make prediction using the trained model
    
    Args:
        employee_data (dict): Employee information
    
    Returns:
        tuple: (prediction_text, confidence, risk_level, risk_percentage)
    """
    try:
        print(f"🎯 Making prediction for employee data...")
        
        # Preprocess input data
        processed_data = preprocess_input(employee_data)
        
        if processed_data is None:
            print("❌ Failed to preprocess data")
            return "Error processing data", 0, "Error", 0
        
        print(f"✅ Data preprocessed successfully, making prediction...")
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        
        print(f"📊 Raw prediction: {prediction}, probabilities: {prediction_proba}")
        
        # Calculate confidence and risk
        confidence = max(prediction_proba) * 100
        risk_percentage = prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
        
        print(f"📈 Confidence: {confidence:.2f}%, Risk percentage: {risk_percentage:.2f}%")
        
        # Determine prediction and risk level based on confidence
        if confidence < 60:  # Low confidence - uncertain prediction
            if prediction == 1:
                prediction_text = "May Leave"
                risk_level = "Medium Risk"
            else:
                prediction_text = "May Stay"
                risk_level = "Medium Risk"
        elif prediction == 1:
            prediction_text = "Likely to Leave"
            risk_level = "High Risk"
        else:
            prediction_text = "Likely to Stay"
            risk_level = "Low Risk"
        
        print(f"🎯 Final prediction: {prediction_text} ({risk_level})")
        return prediction_text, confidence, risk_level, risk_percentage
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Error making prediction", 0, "Error", 0

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/services')
def services():
    """Services page"""
    return render_template('services.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    
    try:
        print(f"🚀 Prediction request received")
        
        if model is None:
            print("❌ Model not loaded")
            return render_template('result.html', 
                                prediction="Error", 
                                confidence=0, 
                                risk_level="Error",
                                risk_percentage=0,
                                employee_data={},
                                error="Model not loaded")
        
        # Get form data
        data = request.form.to_dict()
        print(f"📝 Form data received: {list(data.keys())}")
        
        # Convert numeric fields
        numeric_fields = ['Age', 'Years at Company', 'Monthly Income', 'Number of Dependents', 'Number of Promotions', 'Company Tenure', 'Distance from Home']
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    data[field] = float(data[field])
                    print(f"✅ Converted {field}: {data[field]}")
                except ValueError:
                    print(f"❌ Invalid value for {field}: {data[field]}")
                    return render_template('result.html', 
                                        prediction="Error", 
                                        confidence=0, 
                                        risk_level="Error",
                                        risk_percentage=0,
                                        employee_data=data,
                                        error=f"Invalid value for {field}")
        
        print(f"✅ All numeric fields converted successfully")
        
        # Preprocess input data
        df = preprocess_input(data)
        if df is None:
            print("❌ Preprocessing failed")
            return render_template('result.html', 
                                prediction="Error", 
                                confidence=0, 
                                risk_level="Error",
                                risk_percentage=0,
                                employee_data=data,
                                error="Error preprocessing data")
        
        # Make prediction
        prediction_text, confidence, risk_level, risk_percentage = make_prediction(data)
        
        # Check for errors
        if prediction_text.startswith("Error"):
            print(f"❌ Prediction error: {prediction_text}")
            return render_template('result.html', 
                                prediction="Error", 
                                confidence=0, 
                                risk_level="Error",
                                risk_percentage=0,
                                employee_data=data,
                                error=prediction_text)
        
        # Save prediction to database (temporarily disabled for performance testing)
        # try:
        #     save_prediction({
        #         'timestamp': datetime.now(),
        #         'prediction': prediction_text,
        #         'confidence': confidence,
        #         'risk_level': risk_level,
        #         'risk_percentage': risk_percentage,
        #         'employee_data': data
        #     })
        # except Exception as e:
        #     print(f"Warning: Could not save to database: {e}")
        
        return render_template('result.html', 
                            prediction=prediction_text,
                            confidence=confidence,
                            risk_level=risk_level,
                            risk_percentage=risk_percentage,
                            employee_data=data)
                            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', 
                            prediction="Error", 
                            confidence=0, 
                            risk_level="Error",
                            risk_percentage=0,
                            employee_data=data if 'data' in locals() else {},
                            error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Get the actual feature names from the model
        model_feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_columns
        
        # Ensure correct column order
        df = df[model_feature_names]
        
        # Preprocess data
        for col in categorical_columns:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                unique_values = le.classes_
                df[col] = df[col].apply(lambda x: x if x in unique_values else unique_values[0])
                df[col] = le.transform(df[col])
        
        if numerical_columns:
            df[numerical_columns] = scaler.transform(df[numerical_columns])
        
        # Make prediction
        prediction_text, confidence, risk_level, risk_percentage = make_prediction(data)
        
        # Check for errors
        if prediction_text.startswith("Error"):
            return jsonify({'error': prediction_text}), 500
        
        result = {
            'prediction': prediction_text,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_percentage': risk_percentage
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def api_get_predictions():
    """API endpoint to get all predictions"""
    try:
        predictions = get_predictions()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Display prediction statistics"""
    try:
        stats = mongo_client.get_prediction_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'error': 'Error getting statistics'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('index.html'), 500

def main():
    """Main function to run the Flask application"""
    print("Starting Employee Attrition Predictor...")
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting...")
        return
    
    # Test MongoDB connection (optional)
    try:
        mongo_client.get_prediction_stats()
        print("MongoDB connected")
    except:
        print("MongoDB not available")
    
    print("Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 