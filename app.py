from flask import Flask, request, render_template_string
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import joblib

# Import the Perceptron class so that joblib.load can deserialize the object
from src.model import Perceptron

app = Flask(__name__)

# Path to the saved model (ensure the file is saved as a pickle file)
MODEL_PATH = "model/perceptron_model.pkl"


def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        raise FileNotFoundError(
            "Model file not found. Please train and save the model as model/perceptron_model.pkl."
        )


def plot_decision_boundary(model, point=None):
    # Define plot range; adjust these values based on your synthetic data
    x_min, x_max = -2, 4
    y_min, y_max = -1, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions for the grid points
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

    # If a point is provided, highlight it
    if point is not None:
        plt.scatter(point[0], point[1], color='green', marker='o', s=100,
                    edgecolor='black', label='Input Point')
        plt.legend()

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary with Input Point")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64


@app.route('/', methods=['GET'])
def index():
    html_form = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
      <title>Perceptron Prediction</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
          body { background-color: #f8f9fa; }
          .card { margin-top: 50px; }
          h1 { color: #343a40; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="card mx-auto" style="max-width: 500px;">
          <div class="card-body">
            <h1 class="card-title text-center">Perceptron Prediction</h1>
            <form action="/predict" method="post">
              <div class="form-group">
                <label for="feature1">Feature 1 (x):</label>
                <input type="text" class="form-control" id="feature1" name="feature1" required>
              </div>
              <div class="form-group">
                <label for="feature2">Feature 2 (y):</label>
                <input type="text" class="form-control" id="feature2" name="feature2" required>
              </div>
              <button type="submit" class="btn btn-primary btn-block">Predict</button>
            </form>
          </div>
        </div>
      </div>
    </body>
    </html>
    '''
    return render_template_string(html_form)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and process form data
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        input_point = np.array([feature1, feature2]).reshape(1, -1)

        # Load the trained model
        model = load_model()

        # Get the prediction from the model
        prediction = model.predict(input_point)[0]

        # Generate decision boundary plot with the input point marked
        plot_img = plot_decision_boundary(model, point=[feature1, feature2])

        html_response = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
          <title>Prediction Result</title>
          <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
          <style>
              body {{ background-color: #f8f9fa; }}
              .card {{ margin-top: 50px; }}
              h1 {{ color: #343a40; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="card mx-auto" style="max-width: 600px;">
              <div class="card-body">
                <h1 class="card-title text-center">Prediction Result</h1>
                <p class="text-center">Input Point: ({feature1}, {feature2})</p>
                <p class="text-center">Predicted Target: <strong>{prediction}</strong></p>
                <div class="text-center">
                  <img src="data:image/png;base64,{plot_img}" class="img-fluid" alt="Decision Boundary Plot">
                </div>
                <div class="text-center mt-3">
                  <a href="/" class="btn btn-secondary">Try Another Input</a>
                </div>
              </div>
            </div>
          </div>
        </body>
        </html>
        '''
        return render_template_string(html_response)
    except Exception as e:
        error_html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
          <title>Error</title>
          <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
          <style>
              body {{ background-color: #f8f9fa; }}
              .container {{ margin-top: 50px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="alert alert-danger" role="alert">
              <h4 class="alert-heading">Error!</h4>
              <p>{str(e)}</p>
              <hr>
              <a href="/" class="btn btn-danger">Go Back</a>
            </div>
          </div>
        </body>
        </html>
        '''
        return render_template_string(error_html), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
