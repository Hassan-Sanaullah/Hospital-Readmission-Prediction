
# Hospital Readmission Prediction App

Predict whether a patient is likely to be readmitted to the hospital within 30 days using trained machine learning models. Built with Python, scikit-learn, and Streamlit.

---

## **Project Structure**

```
hospital_readmission_app/
│
├── data/
│   └── diabetic_data.csv      # Original dataset
├── models/                    # Folder to store trained models (.pkl)
├── preprocess.py              # Preprocessing functions
├── train_model.py             # Train and save models
├── app.py                     # Streamlit UI app
├── requirements.txt           # Python dependencies
└── README.md
```

---

## **Setup and Installation (Windows)**

### 1. Install Python

* Ensure **Python 3.10+** is installed.
* Check version:

```bash
python --version
```

---

### 2. Clone or copy project folder

* Place all files in a folder, e.g., `C:\hospital_readmission_app`.

---

### 3. Create a virtual environment (recommended)

```bash
cd C:\hospital_readmission_app
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* pandas, numpy
* scikit-learn
* joblib
* streamlit

---

### 5. Train models

* Run `train_model.py` to preprocess data and train models. This will save trained models in the `models/` folder:

```bash
python train_model.py
```

You should see output like:

```
Logistic_Regression saved.
Decision_Tree saved.
Random_Forest saved.
```

---

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

* A local web server will start, and your default browser should open the app.
* Use the **sidebar** to input patient information and select the model.
* Click **Predict Readmission** to see the result and probability.

---

### **7. Notes**

* All categorical features are converted to **one-hot encoding**.
* Decision Tree and Random Forest models now display probabilities as well as predicted class.
* You can adjust numeric sliders, check medications, and select categorical options to see how predictions change.

---

### **8. Stopping the app**

* Press `CTRL + C` in the terminal to stop Streamlit.

---

### **Optional**

* You can tweak the probability threshold in `app.py` for the decision tree or random forest if you want to adjust sensitivity.

---

This README provides **all instructions needed to run the app locally on Windows**.
