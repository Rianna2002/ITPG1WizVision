# Singapore Rental Prediction System

This application predicts monthly rental prices for Singapore properties using machine learning models like XGBoost and Multi-Layer Perceptrons. It includes MLflow for experiment tracking and is containerized with Docker for easy deployment.

## Features

- **Rental Prediction**: Predict monthly rent based on property details (town, flat type, etc.)
- **Model Training**: Train and compare multiple ML models (XGBoost, MLP)
- **Model Tracking**: Track experiments, parameters, and metrics with MLflow
- **User Interface**: Simple and intuitive Streamlit interface
- **Containerized**: Easily deployable with Docker

## Project Structure

```
rental-prediction/
├── data/
│   └── RentingOutofFlats2025.csv
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model_trainer.py
│   ├── predictor.py
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```
## Data Source
https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view

## Setup and Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rental-prediction.git
   cd rental-prediction
   ```

2. Place your `RentingOutofFlats2025.csv` data file in the `data/` directory.

3. Build the Docker image:
   ```bash
   docker build -t rental-prediction .
   ```

4. Run the Docker container:
   ```bash
   docker run -p 8501:8501 -p 5000:5000 rental-prediction
   ```

5. Access the application:
   - Streamlit UI: http://localhost:8501
   - MLflow UI: http://localhost:5000

### Option 2: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rental-prediction.git
   cd rental-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your `RentingOutofFlats2025.csv` data file in the `data/` directory.

5. Start the MLflow server:
   ```bash
   mlflow ui --host 0.0.0.0
   ```

6. In a new terminal, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

7. Access the application:
   - Streamlit UI: http://localhost:8501
   - MLflow UI: http://localhost:5000

## Usage

1. **Train a Model**:
   - Go to the "Train Model" page
   - Select the dataset
   - Choose a model type (XGBoost or MLP)
   - Click "Train Model"

2. **Make Predictions**:
   - Go to the "Predict" page
   - Enter property details (town, flat type, etc.)
   - Click "Predict Rent"

3. **View Model Performance**:
   - Go to the "Model Performance" page
   - Compare metrics across different models
   - View detailed parameters and visualizations

## Docker Details

The included Dockerfile:
- Uses Python 3.9 as the base image
- Installs all required dependencies
- Sets up MLflow and Streamlit
- Exposes ports 8501 (Streamlit) and 5000 (MLflow)
- Configures an entrypoint script to run both services

## Model Details

### XGBoost
- Gradient boosting framework
- Well-suited for tabular data
- Effective for regression tasks like rent prediction

### Multi-Layer Perceptron (MLP)
- Neural network architecture
- Can capture complex non-linear relationships
- Includes dropout layers to prevent overfitting

## MLflow Integration

The application uses MLflow to:
- Track experiments
- Log hyperparameters
- Record performance metrics
- Save and load models
- Visualize results

## Data Requirements

The application expects a CSV file with the following columns:
- `rent_approval_date`
- `town`
- `block`
- `street_name`
- `flat_type`
- `monthly_rent`