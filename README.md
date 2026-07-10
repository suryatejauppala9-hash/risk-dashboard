# Portfolio Risk Analytics Dashboard

An institutional-grade risk management and portfolio analytics dashboard built with **Python**, **Streamlit**, and **Plotly**. This application allows investors to analyze historical returns, evaluate downside risk (VaR/CVaR), stress-test allocations against historical crises, optimize weights, and get AI-powered portfolio commentary tailored to retail and Indian market contexts.

---

## Features

### 1. Core Portfolio Performance
* **Historical Performance:** Analyzes total returns, annualized returns, annualized volatility, positive days, and best/worst trading days.
* **Benchmarking:** Side-by-side comparison against benchmarks like **NIFTY 50**, **S&P 500**, **NASDAQ**, and **Sensex**.
* **Visualizations:** Interactive charts for cumulative returns, daily returns, rolling annual returns, and simulated portfolio value.

### 2. Risk & Downside Analytics
* **Drawdown Deep-Dive:** Computes maximum drawdowns, average drawdown durations, and displays a timeline/table of top drawdown periods.
* **Tail-Risk (VaR & CVaR):** Estimates **Value at Risk (VaR)** and **Expected Shortfall (CVaR / CVaR)** using three methodologies:
  * **Historical** (empirical distribution)
  * **Parametric** (variance-covariance model)
  * **Monte Carlo Simulation** (interactive path generation with customizable confidence levels and time horizons)
* **Risk Adjustments:** Calculates **Sharpe**, **Sortino**, and **Calmar** ratios.

### 3. Correlation & Diversification
* **Pairwise Correlation:** Generates an interactive correlation heatmap and tracks rolling correlation between specific asset pairs.
* **Risk Contribution:** Measures the actual marginal risk contributed by each asset relative to its allocation weight.
* **Diversification Ratio:** Automatically determines whether your portfolio structure is effectively mitigating volatility.

### 4. Stress Testing & Optimization
* **Historical Shocks:** Simulates portfolio impact under historic market crises (e.g., 2008 Financial Crisis, 2020 COVID Crash, High Inflation).
* **Custom Shocks:** Allows you to input custom positive/negative shocks to evaluate vulnerability.
* **Modern Portfolio Theory (MPT):** Generates and plots the **Efficient Frontier**, identifying the **Minimum Variance** and **Maximum Sharpe** portfolios for rebalancing.

### 5. GenAI Risk Advisor
* Integrates **Gemini 3.1 Flash Lite** to deliver direct, punchy risk desk commentary. It flags concentration, tail risk, and benchmark vulnerabilities without generic disclaimers.

---

## Setup & Installation

### Prerequisites
* Python 3.10+
* A Gemini API key (from [Google AI Studio](https://aistudio.google.com/))

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suryatejauppala9-hash/risk-dashboard.git
   cd risk-dashboard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## Project Structure
* `dashboard/` - Frontend components, pages, and interactive Plotly configurations.
* `analytics/` - Core mathematical libraries (VaR, drawdowns, optimizations, AI advisor integration).
* `data/` - Cached historical pricing logic.
* `tests/` - Unit tests for quantitative calculations.
