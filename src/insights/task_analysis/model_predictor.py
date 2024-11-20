import os
import numpy as np
import joblib

class PlanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.le = None
        self.models_dir = os.path.join('src', 'insights', 'models')
        
    def load_model(self):
        """Load the trained model and its components"""
        try:
            model_path = os.path.join(self.models_dir, 'mobile_plan_model.joblib')
            scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
            le_path = os.path.join(self.models_dir, 'label_encoder.joblib')
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.le = joblib.load(le_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_plan(self, data, minutes, sms):
        """Predict the best plan based on usage"""
        if not all([self.model, self.scaler, self.le]):
            if not self.load_model():
                return "Error: Could not load model"
        
        try:
            input_data = np.array([[data, minutes, sms]])
            scaled_input = self.scaler.transform(input_data)
            prediction = self.model.predict(scaled_input)
            plan_name = self.le.inverse_transform(prediction)
            return plan_name[0]
        
        except Exception as e:
            return f"Error making prediction: {e}"
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not all([self.model, self.scaler, self.le]):
            return "Model not loaded"
        
        return {
            "model_type": type(self.model).__name__,
            "model_path": os.path.join(self.models_dir, 'mobile_plan_model.joblib'),
            "is_loaded": all([self.model, self.scaler, self.le])
        }


def predict_single_plan(data, minutes, sms):
    """Utility function for quick single prediction"""
    predictor = PlanPredictor()
    return predictor.predict_plan(data, minutes, sms)

def batch_predict_plans(usage_list):
    """Utility function for batch predictions"""
    predictor = PlanPredictor()
    predictions = []
    for data, minutes, sms in usage_list:
        prediction = predictor.predict_plan(data, minutes, sms)
        predictions.append(prediction)
    return predictions 