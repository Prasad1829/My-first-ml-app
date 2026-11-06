# Restaurant Rating Prediction App

A small Streamlit machine-learning app that estimates a restaurant's aggregate rating from its average cost for two, table-booking availability, online-delivery availability, and price range. The repository includes the Streamlit interface, a restaurant dataset, a notebook for the prediction project, and serialized model/scaler artifacts.

## Features

- Interactive form for average cost, table booking, online delivery, and price range (1–4).
- Converts booking and delivery choices to binary values and scales the inputs before prediction.
- Presents the predicted rating as **Bad**, **Average**, **Good**, or **Excellent**.
- Uses the dataset file `Dataset .csv` and includes the notebook `Restaurant_rating_Prediction.ipynb` for project exploration.

## Repository contents

| File | Description |
| --- | --- |
| `first-app.py` | Streamlit application and prediction interface. |
| `Restaurant_rating_Prediction.ipynb` | Notebook for exploring the restaurant-rating prediction project. |
| `Dataset .csv` | Restaurant data, including cost, booking/delivery options, price range, aggregate rating, and votes. |
| `Restaurant_rating.pkl` | Serialized feature scaler loaded by the application. |
| `Rating_predict.pkl` | Serialized prediction model loaded by the application. |
| `requirements.txt` | Python dependencies for running the app. |
| `.devcontainer/devcontainer.json` | Python 3.11 development-container configuration. |

## Requirements

- Python 3.11 or a compatible Python version
- The two model artifacts (`Restaurant_rating.pkl` and `Rating_predict.pkl`) in the project root

## Run the app

From the repository root, create and activate a virtual environment, install dependencies, and start Streamlit:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
streamlit run first-app.py
```

Streamlit will print a local URL (usually http://localhost:8501) to open in your browser.

## How prediction works

The app accepts four inputs: `Average Cost for two`, `Has Table booking`, `Has Online delivery`, and `Price range`. It encodes the Yes/No options as 1/0, passes the resulting features through the saved scaler, and calls the saved model to predict a rating. The displayed category is based on the prediction thresholds in `first-app.py`:

- Below 2.5: **Bad**
- 2.5 to below 3.5: **Average**
- 3.5 to below 4.0: **Good**
- 4.0 or above: **Excellent**

## Notebook

Open `Restaurant_rating_Prediction.ipynb` in Jupyter Notebook or JupyterLab to explore the project. The notebook currently references the dataset through a machine-specific absolute path; update its `pd.read_csv(...)` path to the location of `Dataset .csv` before running it in another environment. The notebook also imports Matplotlib and Seaborn, which are not listed in the app's `requirements.txt`; install them separately if needed for notebook work:

```bash
python -m pip install matplotlib seaborn jupyter
```

## Notes

- Run the app from the repository root so its relative paths to the `.pkl` model files resolve.
- The notebook and the Streamlit app have different dependency needs; `requirements.txt` covers the app dependencies.
