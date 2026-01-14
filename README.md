# 🔧 Predictive Maintenance using LSTM (Remaining Useful Life Prediction)

Predictive maintenance aims to anticipate equipment failure before it occurs, enabling timely maintenance and reducing unplanned downtime.  
This project implements an **end-to-end deep learning–based predictive maintenance system** to estimate the **Remaining Useful Life (RUL)** of industrial machines using multivariate time-series sensor data.

An **LSTM (Long Short-Term Memory) neural network** is used to capture temporal degradation patterns from sensor readings. The system also includes an **interactive Streamlit web application** that allows users to upload sensor data and obtain real-time RUL predictions.

---

## 📌 Key Features
- Time-series modeling of sensor data using **LSTM**
- Sliding-window sequence generation for temporal learning
- Regression-based RUL prediction
- Interactive **web interface** for predictions
- Downloadable prediction results for further analysis
- Designed with **scalability and cloud deployment** in mind

---

## 🧠 Project Overview

Industrial machines generate large volumes of sensor data during operation. Predictive maintenance leverages this data to forecast failures in advance.

In this project:
- Historical sensor data is processed and structured into time-series sequences
- An LSTM model is trained to predict the Remaining Useful Life (RUL) of each machine
- A user-friendly interface enables easy experimentation and inference

This approach supports **data-driven maintenance planning** and can significantly reduce maintenance costs and operational risks.

---

## 🛠 Technologies Used

- **Python 3**
- **TensorFlow / Keras** – Deep learning model development
- **Scikit-learn** – Data preprocessing and scaling
- **Pandas & NumPy** – Data manipulation
- **Streamlit** – Web application interface
- **Matplotlib** – Visualization

---


## ⚙️ Setup Instructions

### Prerequisites
- Python 3.x
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt

