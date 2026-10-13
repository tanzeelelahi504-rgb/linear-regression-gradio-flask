import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import gradio as gr
from flask import Flask, request, render_template_string
from threading import Thread


def load_data(path):
    df = pd.read_csv(path.strip('"'))
    print(df.head())
    return df


def clean_data(df):
    print("Null values:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def train_model(df, features, target):
    x = df[features].values
    y = df[target].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print("R2 score:", r2_score(y_test, pred))
    return model, y_test, pred


def show_graphs(df, feature, target, y_test, pred):
    plt.scatter(df[feature], df[target])
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(feature + " vs " + target)
    plt.show()

    plt.scatter(y_test, pred)
    plt.xlabel("Actual " + target)
    plt.ylabel("Predicted " + target)
    plt.title("Actual vs Predicted")
    plt.show()


path1 = r"C:\Users\Rauf\Desktop\ann\house_prices.csv"

df1 = load_data(path1)
df1 = clean_data(df1)

features1 = ["area_sqft", "bedrooms", "age_years", "distance_km"]
model1, y1_test, pred1 = train_model(df1, features1, "price")
show_graphs(df1, "area_sqft", "price", y1_test, pred1)


path2 = r"C:\Users\Rauf\Desktop\ann\diabetes.csv"

df2 = load_data(path2)
df2 = clean_data(df2)

features2 = ["age", "bmi", "bp", "s5"]
model2, y2_test, pred2 = train_model(df2, features2, "target")
show_graphs(df2, "bmi", "target", y2_test, pred2)


def predict_house(area, bedrooms, age, distance):
    price = model1.predict([[area, bedrooms, age, distance]])[0]
    return "Predicted price: $" + str(round(price))


def predict_diabetes(age, bmi, bp, s5):
    score = model2.predict([[age, bmi, bp, s5]])[0]
    return "Predicted score: " + str(round(score, 2))


house_ui = gr.Interface(
    fn=predict_house,
    inputs=[gr.Number(label="Area (sq ft)", value=1800),
            gr.Number(label="Bedrooms", value=3),
            gr.Number(label="House age (years)", value=15),
            gr.Number(label="Distance from city (km)", value=10)],
    outputs="text")

diabetes_ui = gr.Interface(
    fn=predict_diabetes,
    inputs=[gr.Number(label="Age", value=50),
            gr.Number(label="BMI", value=26),
            gr.Number(label="Blood pressure", value=95),
            gr.Number(label="s5", value=4.6)],
    outputs="text")

demo = gr.TabbedInterface([house_ui, diabetes_ui], ["House Price", "Diabetes"])
demo.launch()


app = Flask(__name__)

page = """
<h1>Linear Regression</h1>

<h2>House Price</h2>
<form method="POST" action="/house">
    <input name="area" placeholder="Area (sq ft)"><br>
    <input name="bedrooms" placeholder="Bedrooms"><br>
    <input name="age" placeholder="House age"><br>
    <input name="distance" placeholder="Distance (km)"><br>
    <button>Predict</button>
</form>

<h2>Diabetes</h2>
<form method="POST" action="/diabetes">
    <input name="age" placeholder="Age"><br>
    <input name="bmi" placeholder="BMI"><br>
    <input name="bp" placeholder="Blood pressure"><br>
    <input name="s5" placeholder="s5 (example 4.6)"><br>
    <button>Predict</button>
</form>

<h2>{{ result }}</h2>
"""


@app.route("/")
def home():
    return render_template_string(page, result="")


@app.route("/house", methods=["POST"])
def house_result():
    a = float(request.form["area"])
    b = float(request.form["bedrooms"])
    c = float(request.form["age"])
    d = float(request.form["distance"])
    price = model1.predict([[a, b, c, d]])[0]
    return render_template_string(page, result="Predicted price: $" + str(round(price)))


@app.route("/diabetes", methods=["POST"])
def diabetes_result():
    a = float(request.form["age"])
    b = float(request.form["bmi"])
    c = float(request.form["bp"])
    d = float(request.form["s5"])
    score = model2.predict([[a, b, c, d]])[0]
    return render_template_string(page, result="Predicted score: " + str(round(score, 2)))


def run_flask():
    app.run(port=5000, use_reloader=False)


Thread(target=run_flask).start()