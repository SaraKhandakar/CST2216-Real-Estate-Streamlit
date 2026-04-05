from src.data_loader import load_data
from src.preprocess import split_data
from src.train import train_models
from src.predict import save_model

DATA_PATH = "data/final.csv"
MODEL_PATH = "models/real_estate_model.pkl"
TARGET_COLUMN = "price"

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Splitting data...")
    X, y, x_train, x_test, y_train, y_test = split_data(df, TARGET_COLUMN)

    print("Training models...")
    results = train_models(x_train, y_train, x_test, y_test)

    best_model_name = min(results, key=lambda k: results[k]["mae"])
    best_model = results[best_model_name]["model"]
    best_mae = results[best_model_name]["mae"]

    print(f"Best model: {best_model_name}")
    print(f"MAE: {best_mae}")

    save_model(best_model, MODEL_PATH)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()