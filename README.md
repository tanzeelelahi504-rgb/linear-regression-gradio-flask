# Linear Regression Project with Gradio and Flask

This project uses **Linear Regression** on two different datasets and makes two user interfaces (Gradio and Flask) to predict values. It runs in Jupyter Notebook.

## Datasets

| File | Rows | What we predict | Input columns used |
|------|------|-----------------|--------------------|
| `house_prices.csv` | 500 | House price (`price`) | `area_sqft`, `bedrooms`, `age_years`, `distance_km` |
| `diabetes.csv` | 442 | Disease score (`target`) | `age`, `bmi`, `bp`, `s5` |

Note: `house_prices.csv` is a generated (sample) dataset. `diabetes.csv` is the real diabetes dataset.

## Files

- `linear_regression_gradio_flask.ipynb` - main notebook (all the code)
- `house_prices.csv` - dataset 1
- `diabetes.csv` - dataset 2
- `README.md` - this file

## Libraries Used

- pandas
- matplotlib
- scikit-learn
- gradio
- flask

Install them with:

```
pip install gradio flask scikit-learn pandas matplotlib
```

## How to Run

1. Save both CSV files in one folder (for example Downloads).
2. Open `linear_regression_gradio_flask.ipynb` in Jupyter Notebook.
3. Right click each CSV file -> **Copy as path**, then paste it in `path1` and `path2`:
   ```python
   path1 = r"C:\Users\Faisal\Downloads\house_prices.csv"
   path2 = r"C:\Users\Faisal\Downloads\diabetes.csv"
   ```
4. Run all cells from top to bottom (Cell -> Run All).
5. For the Gradio app, use the link or the app shown under the Gradio cell.
6. For the Flask app, open `http://127.0.0.1:5000` in your browser.

## Steps in the Project

1. Load the data from the CSV file
2. Check null values and duplicate rows, then remove them
3. Split the data into 80% training and 20% testing
4. Train the Linear Regression model
5. Check the model with the R2 score
6. Show graphs with matplotlib
7. Make the Gradio user interface
8. Make the Flask user interface

## Functions in the Notebook

| Function | Work |
|----------|------|
| `load_data(path)` | reads the CSV file |
| `clean_data(df)` | checks and removes null and duplicate values |
| `train_model(df, features, target)` | trains the model and prints the R2 score |
| `show_graphs(...)` | shows the matplotlib graphs |
| `predict_house(...)` | Gradio function for house price |
| `predict_diabetes(...)` | Gradio function for diabetes score |
| `home()` | Flask home page |
| `house_result()` | Flask house price result |
| `diabetes_result()` | Flask diabetes result |
| `run_flask()` | starts the Flask server |

## Graphs

For each dataset there are two graphs:
- Feature vs target (`area_sqft` vs `price` and `bmi` vs `target`)
- Actual vs predicted values

## Results

| Dataset | R2 score |
|---------|----------|
| House prices | about 0.985 |
| Diabetes | about 0.457 |

The house price score is very high because the sample data follows a clear pattern. The diabetes score is lower because real data is more mixed.

## Example Inputs

**House price:** area = 1800, bedrooms = 3, age = 15, distance = 10

**Diabetes:** age = 50, bmi = 26, bp = 95, s5 = 4.6

## Problems and Solutions

- **Flask link not clickable:** the red text in Jupyter is normal. Copy `http://127.0.0.1:5000` into the browser.
- **Port already in use:** restart the kernel and run all cells again, or change `port=5000` to `port=5001`.
- **File not found error:** check that the path in `path1` and `path2` is correct.
