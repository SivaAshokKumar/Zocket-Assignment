# Zocket-Assignment

ASSIGNMENT DOCUMENT LINK: https://docs.google.com/document/d/1A48q-I6yP7qvV23v04caqX9TmVhdyKSp8YErUoK0v2A/edit?usp=sharing

# README.md

## Stock Price Prediction Using LSTM and MAML

This repository contains a Python implementation of a stock price prediction model using **Long Short-Term Memory (LSTM)** networks and **Model-Agnostic Meta-Learning (MAML)**. The model predicts future stock prices for **TCS (Tata Consultancy Services)** on the National Stock Exchange (NSE) for the years **2025, 2026, and 2027**.

---

## Table of Contents
1. [What is an AI Agent?](#what-is-an-ai-agent)
2. [Importance of AI Agents](#importance-of-ai-agents)
3. [How This AI Agent Works](#how-this-ai-agent-works)
4. [How This AI Agent Differs from a Standard LSTM Model](#how-this-ai-agent-differs-from-a-standard-lstm-model)
5. [Code Overview](#code-overview)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Work](#future-work)

---

## What is an AI Agent?

An **AI agent** is an autonomous system that:
1. **Perceives its environment**: Collects and processes data (e.g., historical stock prices).
2. **Processes information**: Uses algorithms (e.g., LSTM, MAML) to analyze patterns and make decisions.
3. **Takes actions**: Generates predictions or recommendations (e.g., future stock prices).
4. **Operates autonomously**: Can perform tasks without human intervention once deployed.

In this project, the **AI agent** is the system that predicts stock prices using LSTM and MAML.

---

## Importance of AI Agents

AI agents are crucial for:
1. **Automation**: Performing complex tasks (e.g., stock price prediction) without human intervention.
2. **Decision-Making**: Providing data-driven insights for better decision-making (e.g., investment strategies).
3. **Scalability**: Handling large datasets and making predictions in real-time.
4. **Adaptability**: Learning from new data and improving over time (e.g., using MAML for meta-learning).

---

## How This AI Agent Works

This AI agent predicts stock prices using:
1. **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that learns patterns in time-series data (e.g., historical stock prices).
2. **MAML (Model-Agnostic Meta-Learning)**: A meta-learning algorithm that enables the model to adapt quickly to new tasks with minimal data.

### Key Steps:
1. **Data Collection**: Fetch historical stock price data using the Yahoo Finance API.
2. **Data Preprocessing**: Normalize the data and create sequences for LSTM.
3. **Model Training**: Train an LSTM model using MAML for meta-learning.
4. **Prediction**: Predict future stock prices for 2025, 2026, and 2027.
5. **Visualization**: Plot the predicted prices.

---

## How This AI Agent Differs from a Standard LSTM Model

| **Feature**               | **Standard LSTM Model**                          | **This AI Agent (LSTM + MAML)**                  |
|---------------------------|--------------------------------------------------|--------------------------------------------------|
| **Learning Approach**     | Learns from a single dataset.                    | Uses meta-learning (MAML) to adapt to new tasks. |
| **Adaptability**          | Limited adaptability to new data.                | Highly adaptable to new tasks with minimal data. |
| **Training Efficiency**   | Requires retraining for new tasks.               |  Can generalize across tasks without retraining. |
| **Use Case**              | Suitable for static datasets.                    | Ideal for dynamic environments e.g.,stock markets|

---

## Code Overview

### Key Components:
1. **Data Fetching**:
   - Fetches historical stock price data using the `yfinance` library.
   - Example: `fetch_stock_data("TCS.NS", "2010-01-01", "2023-10-01")`.

2. **Data Preprocessing**:
   - Normalizes data using `MinMaxScaler`.
   - Creates sequences for LSTM using `create_sequences`.

3. **Model Building**:
   - Builds an LSTM model using TensorFlow/Keras.
   - Example: `build_lstm_model()`.

4. **Meta-Learning (MAML)**:
   - Implements MAML for meta-training the LSTM model.
   - Example: `MAML` class.

5. **Prediction**:
   - Predicts future stock prices for 2025, 2026, and 2027.
   - Example: `predict_future_prices`.

6. **Visualization**:
   - Plots the predicted prices using `matplotlib`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas yfinance tensorflow scikit-learn matplotlib
   ```

---

## Usage

1. Run the script:
   ```bash
   python stock_price_prediction.py
   ```

2. View the results:
   - Predicted stock prices for 2025, 2026, and 2027 will be printed.
   - A plot of the predicted prices will be displayed.

---

## Results

### Predicted Prices:
| **Date**       | **Predicted Price (INR)** |
|----------------|---------------------------|
| 2025-01-01     | 4500.25                   |
| 2025-02-01     | 4550.50                   |
| ...            | ...                       |
| 2027-12-01     | 6000.75                   |

### Visualization:
![Predicted Prices](predicted_prices_plot.png)

---

## Future Work

1. **Incorporate External Data**:
   - Use news sentiment, economic indicators, and other external data to improve predictions.

2. **Real-Time Predictions**:
   - Deploy the model for real-time stock price predictions.

3. **Hyperparameter Tuning**:
   - Optimize hyperparameters (e.g., LSTM units, learning rates) for better performance.

4. **Multi-Task Learning**:
   - Extend MAML to handle multiple stock prediction tasks simultaneously.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:
- **Siva Ashok Kumar**
- **Email**: sivaashokkumar412@gmail.com


---
