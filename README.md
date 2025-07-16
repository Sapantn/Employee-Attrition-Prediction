# Employee Attrition Predictor

A complete, production-ready website that predicts employee attrition using XGBoost machine learning model.

## 🚀 Features

- **Interactive Website**: Full-fledged web application with homepage and results page
- **Machine Learning Model**: XGBoost classifier trained on HR dataset
- **Real-time Predictions**: Instant attrition predictions based on employee data
- **Database Storage**: MongoDB integration for storing all predictions
- **Responsive Design**: Clean, modern UI with proper styling
- **Form Validation**: Client-side validation for better user experience

## 📁 Project Structure

```
employee-attrition-predictor/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── train.py
│   │   └── xgb_model.pkl
│   └── db/
│       └── mongo_client.py
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│
├── requirements.txt
└── README.md
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB (local installation or MongoDB Atlas)
- pip

### Installation

1. **Clone or download the project**
   ```bash
   cd employee-attrition-predictor
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**
   - Install MongoDB locally, or
   - Use MongoDB Atlas (cloud service)
   - Ensure MongoDB is running on `localhost:27017`

4. **Add your HR dataset**
   - Place your training data in `data/train.csv`
   - Place your test data in `data/test.csv` (optional)
   - Ensure the dataset has an `Employee ID` column and `Attrition` target column

5. **Train the model**
   ```bash
   cd backend
   python model/train.py
   ```

6. **Run the website**
   ```bash
   python app.py
   ```

7. **Access the website**
   - Open `http://localhost:5000` in your browser

## 🎯 Usage

1. **Homepage**: Visit the main page with the employee information form
2. **Fill Form**: Complete all required employee fields
3. **Submit**: Click "Predict Attrition" to get the prediction
4. **Results**: View the prediction result and submitted data
5. **Database**: All predictions are automatically stored in MongoDB

## 🔧 Website Features

### Homepage (`/`)
- Interactive form with all employee fields
- Real-time form validation
- Clean, responsive design
- Professional styling

### Results Page (`/predict`)
- Displays prediction result ("Likely to Stay" or "Likely to Leave")
- Shows all submitted employee data
- Option to return to homepage for new predictions

## 🗄️ Database Schema

**Database**: `attrition_db`
**Collection**: `predictions`

**Document Structure:**
```json
{
  "timestamp": "2023-12-01T10:30:00Z",
  "employee_data": {
    "Age": 30,
    "Gender": "Male",
    "Years at Company": 4,
    "Job Role": "HR",
    "Monthly Income": 6200,
    "Work-Life Balance": "Good",
    "Job Satisfaction": "High",
    "Performance Rating": "Above Average",
    "Number of Promotions": 1,
    "Overtime": "No",
    "Distance from Home": 12,
    "Education Level": "Bachelor",
    "Marital Status": "Married",
    "Number of Dependents": 2,
    "Job Level": "Mid",
    "Company Size": "Medium",
    "Company Tenure": 48,
    "Remote Work": "Yes",
    "Leadership Opportunities": "Yes",
    "Innovation Opportunities": "Yes",
    "Company Reputation": "Strong",
    "Employee Recognition": "High"
  },
  "prediction": "Likely to Leave"
}
```

## 🧠 Model Details

- **Algorithm**: XGBoost Classifier
- **Training**: 80/20 train-test split
- **Features**: 22 employee attributes
- **Target**: Binary classification (Stay/Leave)
- **Preprocessing**: Handles missing values and categorical encoding

## 📊 Form Fields

The website collects the following employee information:

- Age
- Gender
- Years at Company
- Job Role
- Monthly Income
- Work-Life Balance
- Job Satisfaction
- Performance Rating
- Number of Promotions
- Overtime
- Distance from Home
- Education Level
- Marital Status
- Number of Dependents
- Job Level
- Company Size
- Company Tenure
- Remote Work
- Leadership Opportunities
- Innovation Opportunities
- Company Reputation
- Employee Recognition

## 🔒 Environment Variables

Create a `.env` file in the backend directory (optional):
```
MONGODB_URI=mongodb://localhost:27017/attrition_db
FLASK_ENV=development
FLASK_DEBUG=True
```

## 🐛 Troubleshooting

1. **MongoDB Connection Error**: Ensure MongoDB is running on localhost:27017
2. **Model Not Found**: Run `python model/train.py` to generate the model file
3. **Port Already in Use**: Change port in `app.py` or kill existing process
4. **Dataset Issues**: Ensure your CSV files are in the correct format with required columns

## 📈 Model Performance

The XGBoost model provides:
- High accuracy on employee attrition prediction
- Feature importance analysis
- Robust handling of categorical variables
- Scalable predictions for real-time use

## 🎨 Design Features

- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, professional appearance
- **User-Friendly**: Intuitive form design and navigation
- **Visual Feedback**: Clear prediction results and data display

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License. 