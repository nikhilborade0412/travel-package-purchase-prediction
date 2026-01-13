
# 🌍 Travel Package Purchase Prediction

*A **Machine Learning web application** that predicts whether a customer will purchase a travel package based on demographic, behavioral, and pitch-related features.
The project uses a trained ML model with preprocessing and an interactive **Streamlit** interface for real-time predictions.*

---

## 📌 Table of Contents

* <a href="#project-overview">Project Overview</a>
* <a href="#problem-statement">Problem Statement</a>
* <a href="#solution-approach">Solution Approach</a>
* <a href="#features-used">Features Used</a>
* <a href="#tech-stack">Tech Stack</a>
* <a href="#project-structure">Project Structure</a>
* <a href="#model--preprocessing">Model & Preprocessing</a>
* <a href="#web-application-streamlit">Web Application (Streamlit)</a>
* <a href="#how-to-run-the-project">How to Run the Project</a>
* <a href="#results">Results</a>
* <a href="#author">Author</a>

---

<h2><a class="anchor" id="project-overview"></a>📖 Project Overview</h2>

Travel companies often struggle to identify customers who are most likely to purchase travel packages after a sales pitch.
This project applies **Machine Learning** to predict customer purchase behavior, helping businesses improve **targeted marketing** and **conversion rates**.

The application allows users to enter customer details and instantly get a prediction using a trained model.

---

<h2><a class="anchor" id="problem-statement"></a>❓ Problem Statement</h2>

Given customer demographic and interaction data, predict whether a customer will **purchase a travel package** (`Yes / No`).

---

<h2><a class="anchor" id="solution-approach"></a>🛠 Solution Approach</h2>

1. Data cleaning and preprocessing
2. Handling categorical and numerical features using `ColumnTransformer`
3. Model training and evaluation
4. Saving trained model and preprocessor
5. Building an interactive Streamlit web app for predictions

---

<h2><a class="anchor" id="features-used"></a>📊 Features Used</h2>

### 🔢 Numerical Features

* Age
* Monthly Income
* Duration of Pitch
* Number of Followups
* Number of Trips
* Preferred Property Star
* Pitch Satisfaction Score
* Number of Persons Visiting
* Number of Children Visiting

### 🧾 Categorical Features

* Gender
* Marital Status
* Occupation
* Type of Contact
* Product Pitched
* Designation
* City Tier
* Passport
* Own Car

---

<h2><a class="anchor" id="tech-stack"></a>🧰 Tech Stack</h2>

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Web Framework:** Streamlit
* **Model Persistence:** Pickle
* **Version Control:** Git & GitHub

---

<h2><a class="anchor" id="project-structure"></a>📁 Project Structure</h2>

```
Travel-Package-Prediction/
│
├── app
│   ├── EDA.py                     # Streamlit EDA application
│   └── ml.py                      # Streamlit prediction app
│
├── data
│   └── traveling_data.csv
│
├── model building
│   └── model_building.py
│
├── jupyter notebook
│   ├── EDA.ipynb
│   ├── model_building.ipynb
│   └── Tourism Domain knowledge.ipynb
│
├── pdf
│   └── Travel_Package_Dataset_Domain_Knowledge.pdf
│
├── pptx
│   └── Boosting_travel_package_sales.pptx
│
├── video
│   ├── EDA video.mp4
│   └── Prediction Video.mp4
│
├── pkl
│   ├── tourism_model.pkl
│   ├── preprocessor.pkl
│   ├── tourism_model.joblib
|   └── preprocessor.joblib
|
├── README.md
└── requirements.txt
```

---

<h2><a class="anchor" id="model--preprocessing"></a>🧠 Model & Preprocessing</h2>

* A machine learning classification model was trained on customer data.
* A `ColumnTransformer` was used to preprocess data:

  * Scaling numerical features
  * Encoding categorical features
* Both the trained model and preprocessor were saved and reused during inference.

---

<h2><a class="anchor" id="web-application-streamlit"></a>🌐 Web Application (Streamlit)</h2>

* Clean, dark-themed UI
* Inputs arranged in a **3-column grid layout**
* Controlled inputs using number fields and dropdowns
* Centered **Predict** button
* Displays prediction result with probability
* Celebration animation for positive predictions 🎉

---

<h2><a class="anchor" id="how-to-run-the-project"></a>▶️ How to Run the Project</h2>

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Travel-Package-Prediction.git
cd Travel-Package-Prediction
```

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

<h2><a class="anchor" id="results"></a>📈 Results</h2>

* Predicts whether a customer will purchase a travel package
* Provides probability score for confidence
* Helps sales teams focus on high-potential customers

---


<h2><a class="anchor" id="author--contact"></a>👤 Author & Contact</h2>

**Name:** Nikhil Borade
**Role:** Data Science & Machine Learning Enthusiast

[**GitHub**](https://github.com/nikhilborade0412)
[**LinkedIn**](http://linkedin.com/in/nikhil-borade0412/)

⭐ *If you find this project useful, consider starring the repository!*
---
