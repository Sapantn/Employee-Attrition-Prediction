from pymongo import MongoClient
from datetime import datetime
import os

class MongoDBClient:
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            # Connect to MongoDB
            self.client = MongoClient('localhost', 27017)
            self.db = self.client['attrition_db']
            self.collection = self.db['predictions']
            print("✅ MongoDB connected successfully")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def insert_prediction(self, employee_data, prediction):
        """
        Insert employee data and prediction into MongoDB
        
        Args:
            employee_data (dict): Employee information
            prediction (str): Prediction result ("Likely to Stay" or "Likely to Leave")
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.collection is None:
                print("❌ MongoDB not connected")
                return False
            
            # Create document to insert
            document = {
                'timestamp': datetime.now(),
                'employee_data': employee_data,
                'prediction': prediction
            }
            
            # Insert into database
            result = self.collection.insert_one(document)
            print(f"✅ Prediction stored in MongoDB with ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store prediction: {e}")
            return False

    def save_prediction(self, prediction_data):
        """
        Save prediction data to MongoDB
        
        Args:
            prediction_data (dict): Complete prediction data including timestamp, prediction, confidence, etc.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.collection is None:
                print("❌ MongoDB not connected")
                return False
            
            # Insert into database
            result = self.collection.insert_one(prediction_data)
            print(f"✅ Prediction stored in MongoDB with ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store prediction: {e}")
            return False

    def get_predictions(self, limit=100):
        """
        Retrieve all predictions from database
        
        Args:
            limit (int): Maximum number of records to return
        
        Returns:
            list: List of prediction documents
        """
        try:
            if self.collection is None:
                print("❌ MongoDB not connected")
                return []
            
            predictions = list(self.collection.find().sort('timestamp', -1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                if '_id' in pred:
                    pred['_id'] = str(pred['_id'])
                if 'timestamp' in pred:
                    pred['timestamp'] = pred['timestamp'].isoformat()
            
            return predictions
            
        except Exception as e:
            print(f"❌ Failed to retrieve predictions: {e}")
            return []

    def get_all_predictions(self, limit=100):
        """
        Retrieve all predictions from database (legacy method)
        
        Args:
            limit (int): Maximum number of records to return
        
        Returns:
            list: List of prediction documents
        """
        return self.get_predictions(limit)

    def get_prediction_stats(self):
        """
        Get basic statistics about predictions
        
        Returns:
            dict: Statistics about predictions
        """
        try:
            if self.collection is None:
                return {}
            
            total_predictions = self.collection.count_documents({})
            stay_predictions = self.collection.count_documents({'prediction': 'Likely to Stay'})
            leave_predictions = self.collection.count_documents({'prediction': 'Likely to Leave'})
            
            return {
                'total_predictions': total_predictions,
                'stay_predictions': stay_predictions,
                'leave_predictions': leave_predictions,
                'stay_percentage': (stay_predictions / total_predictions * 100) if total_predictions > 0 else 0,
                'leave_percentage': (leave_predictions / total_predictions * 100) if total_predictions > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Failed to get prediction stats: {e}")
            return {}

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✅ MongoDB connection closed")

# Create a global instance
mongo_client = MongoDBClient()

# Create convenience functions for direct import
def save_prediction(prediction_data):
    """Convenience function to save prediction data"""
    return mongo_client.save_prediction(prediction_data)

def get_predictions(limit=100):
    """Convenience function to get predictions"""
    return mongo_client.get_predictions(limit) 