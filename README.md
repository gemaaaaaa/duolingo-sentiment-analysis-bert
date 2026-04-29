<h1 align="center">Demo Web Interface<br>Duolingo Review Sentiment Analysis</h1>

This is a demonstration of a sentiment analysis model for Duolingo app reviews, developed for *Kecerdasan Buatan Lanjut* final project. Built using the **BERT (bert-base-multilingual-cased)** architecture, the model was trained on the **Duolingo App User Review Play Store Dataset 2025**, achieving an overall accuracy of **81%**.

## Team

- **Gema Satria Tama**
- **Sahila Amalia**
- **Rangga Firman Ade Syah Putra**
- **Yusuf Fahrudin**

## Dataset

The model was trained using the **[Duolingo App User Review Play Store Dataset 2025](https://www.kaggle.com/datasets/belalakhter/duolingo-app-user-review-play-store-dataset-2025)**.

## Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gemaaaaaa/duolingo-sentiment-analysis-bert.git
   cd duolingo-sentiment-analysis-bert
   ```

2. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run streamlit run app.py
   ```