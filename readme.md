```md
# 🌍 Travel Package Purchase Prediction

A **Machine Learning web application** that predicts whether a customer will purchase a travel package based on demographic, behavioral, and pitch-related features.  
The project uses a trained ML model with preprocessing and an interactive **Streamlit** interface for real-time predictions.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Features Used](#features-used)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model & Preprocessing](#model--preprocessing)
- [Web Application (Streamlit)](#web-application-streamlit)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 📖 Project Overview

Travel companies often struggle to identify customers who are most likely to purchase travel packages after a sales pitch.  
This project applies **Machine Learning** to predict customer purchase behavior, helping businesses improve **targeted marketing** and **conversion rates**.

The application allows users to enter customer details and instantly get a prediction using a trained model.

---

## ❓ Problem Statement

Given customer demographic and interaction data, predict whether a customer will **purchase a travel package** (`Yes / No`).

---

## 🛠 Solution Approach

1. Data cleaning and preprocessing  
2. Handling categorical and numerical features using `ColumnTransformer`  
3. Model training and evaluation  
4. Saving trained model and preprocessor  
5. Building an interactive Streamlit web app for predictions  

---

## 📊 Features Used

### Numerical Features
- Age  
- Monthly Income  
- Duration of Pitch  
- Number of Followups  
- Number of Trips  
- Preferred Property Star  
- Pitch Satisfaction Score  
- Number of Persons Visiting  
- Number of Children Visiting  

### Categorical Features
- Gender  
- Marital Status  
- Occupation  
- Type of Contact  
- Product Pitched  
- Designation  
- City Tier  
- Passport  
- Own Car  

---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Web Framework:** Streamlit  
- **Model Persistence:** Pickle  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

```

Travel-Package-Prediction/
│
├──app
|   ├──EDA.py   # Streamlit application
|   └──ml.py    # Streamlit application
|
├── data
|   └──traveling_data.csv
|
├── model building
|   └── model_building.py
|
├──jupyter notebook
|   ├── EDA.ipynb
|   ├── model_building.ipynb
|   └── Tourisom Domain nkowledge.ipynb
|
├── pdf
|   └──Travel_Package_Dataset_Domain_Knowledge.pdf
│
├── pptx
|   └──Boosting_travel_package_sales.pptx 
|
├── video
|   ├── EDA video.mp4   
|   └── Prediction Video.mp4
|
├── pkl/
│   ├── tourism_model.pkl
│   └── preprocessor.pkl
|
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies

````

---

## 🧠 Model & Preprocessing

- A machine learning classification model was trained on customer data.
- A `ColumnTransformer` was used to preprocess data:
  - Scaling numerical features
  - Encoding categorical features
- Both the trained model and preprocessor were saved and reused during inference.

---

## 🌐 Web Application (Streamlit)

- Clean, dark-themed UI  
- Inputs arranged in a **3-column grid layout**  
- Controlled inputs using number fields and dropdowns  
- Centered **Predict** button  
- Displays prediction result with probability  
- Celebration animation for positive predictions 🎉  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Travel-Package-Prediction.git
cd Travel-Package-Prediction
````

### 2️⃣ Create & Activate Virtual Environment (Optional)

```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run ml.py
```

---

## 📈 Results

* Predicts whether a customer will purchase a travel package
* Provides probability score for confidence
* Helps sales teams focus on high-potential customers

---

## 🚀 Future Improvements

* Feature importance visualization
* Deployment on Streamlit Cloud
* CRM system integration
* User authentication

---

## 👨‍💻 Author

**Nikhil Borade**
Aspiring Data Scientist | Machine Learning Enthusiast

⭐ *If you find this project useful, consider starring the repository!*

```
```
