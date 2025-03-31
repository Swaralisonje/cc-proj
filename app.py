import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64
from flask import Flask, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Load dataset
df = pd.read_csv("processed_medal_predictions.csv")

# Feature selection
X = df[['Sport_Code', 'Total_Country_Medals']]
y = df['Medal_Percentage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred = [min(100, max(0, val)) for val in y_pred]

# Classification report
classification_labels = [1 if val >= 50 else 0 for val in y_test]
predicted_labels = [1 if val >= 50 else 0 for val in y_pred]
class_report = classification_report(classification_labels, predicted_labels, output_dict=False)


def generate_prediction_graph(top_countries, sport_name):
    """Generates a bar chart for predicted medal-winning percentages and returns it as a base64 string."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_countries['Country'], top_countries['Predicted_Percentage'], color='skyblue')
    ax.set_xlabel("Predicted Medal Percentage")
    ax.set_ylabel("Country")
    ax.set_title(f"Top 10 Countries for {sport_name.capitalize()}")
    ax.invert_yaxis()

    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{encoded_img}"


def generate_world_map(top_countries):
    """Generates a world map visualization of predicted medal percentages."""
    fig = go.Figure(data=go.Choropleth(
        locations=top_countries['Country'],
        locationmode='country names',
        z=top_countries['Predicted_Percentage'],
        colorscale='Blues',
        marker_line_color='darkgray',
        marker_line_width=0.5,
    ))
    fig.update_layout(title_text="Predicted Medal Percentages by Country", geo=dict(showcoastlines=True))
    return fig.to_html(full_html=False)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sport_name = request.form["sport_name"].strip().lower()
        sport_code = df[df['Sport'].str.lower() == sport_name]['Sport_Code'].values

        if len(sport_code) == 0:
            return f"""
            <html><head><title>Error</title></head>
            <body><h1>Sport not found! Please check the spelling.</h1>
            <a href='/'>Go Back</a></body></html>
            """

        sport_code = sport_code[0]
        sport_data = df[df['Sport_Code'] == sport_code][['Country', 'Total_Country_Medals']].copy()
        sport_data['Sport_Code'] = sport_code

        # Make predictions
        sport_data['Predicted_Percentage'] = rf_model.predict(sport_data[['Sport_Code', 'Total_Country_Medals']])
        sport_data['Predicted_Percentage'] = sport_data['Predicted_Percentage'].clip(0, 100).round(2)

        top_countries = sport_data.sort_values(by='Predicted_Percentage', ascending=False).head(10)

        # Generate graphs
        prediction_graph = generate_prediction_graph(top_countries, sport_name)
        world_map = generate_world_map(top_countries)

        return f"""
        <html>
        <head>
            <title>Medal Prediction Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; background-color: #0e6655; color: white; }}
                .container {{ max-width: 800px; margin: 50px auto; padding: 20px; background: #16a085; border-radius: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid white; padding: 12px; text-align: center; color: black; }}
                th {{ background-color: #0e6655; color: white; }}
                .classification-report {{ background: white; color: black; padding: 15px; border-radius: 10px; text-align: left; }}
                button {{ background: #0e6655; color: white; padding: 10px; border: none; cursor: pointer; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class='container'>
                <h1>Medal Prediction Results for {sport_name.capitalize()}</h1>
                <h2>Top 10 Countries & Predicted Medal Percentage</h2>
                <table>
                    <tr><th>Country</th><th>Predicted Medal Percentage</th></tr>
                    {''.join(f"<tr><td>{row['Country']}</td><td>{row['Predicted_Percentage']}%</td></tr>" for _, row in top_countries.iterrows())}
                </table>
                <h2>Predicted Medal Winning Percentage</h2>
                <img src='{prediction_graph}' alt='Prediction Graph'>
                <h2>Predicted Medal Percentages by Country</h2>
                {world_map}
                <h2>Classification Report</h2>
                <div class='classification-report'><pre>{class_report}</pre></div>
                <button onclick='history.back()'>Go Back</button>
            </div>
        </body>
        </html>
        """

    return f"""
    <html>
    <head>
        <title>Olympic Medal Prediction</title>
        <style>
                body {{ font-family: Arial, sans-serif; text-align: center; background-color: #0e6655; color: white;width:100%; }}
                .container {{ max-width: 1000px; margin: 50px auto; padding: 40px; background: #16a085; border-radius: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px auto; }}
                th, td {{ border: 1px solid white; padding: 12px; text-align: center; color: black; font-size: 16px; }}
                th {{ background-color: #0e6655; color: white; }}
                .classification-report {{ background: white; color: black; padding: 15px; border-radius: 10px; text-align: left; }}
                button {{ background: #0e6655; color: white; padding: 12px 22px; border: none; cursor: pointer; margin-top: 20px;font-size: 16px;border-radius: 5px; }}
                input {{ padding: 15px; width: 60%; border-radius:5px; border: none; font-size: 14px;text-align:center;color: #000000; }}

        </style>
    </head>
    <body>
        <div class='container'>
            <h1>Olympic Medal Prediction</h1>
            <form method='POST'>
                <label>Enter Sport Name:</label>
                <input type='text' name='sport_name' required>
                <button type='submit'>Predict</button>
            </form>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)
