import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def load_data(path='C:/UCZENIE MASZYNOWE/um-projekt-lato-2025-main/heart.csv'):
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Czyści dane (usuwa duplikaty, sprawdza braki, usuwa zbędne kolumny i normalizuje etykietę)."""
    df = df.drop_duplicates()
    df = df.dropna()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'HeartDisease' in df.columns:
        df = df.rename(columns={'HeartDisease': 'target'})
    return df


def preprocess_data(df):
    """Dzieli dane na X i y, koduje kategorie, skaluje cechy i zwraca podział na train/test."""
    X = df.drop(columns='target')
    y = df['target']

    # Zakodowanie zmiennych kategorycznych (one-hot encoding)
    X_encoded = pd.get_dummies(X)

    # Skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Podział na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Wylicza accuracy, ROC AUC oraz confusion matrix dla modelu."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix)


def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # W tym miejscu osoba 2 i 3 mogą zaimportować swoje modele i wywołać:

    # Model 1 - Logistic Regression
    model1 = LogisticRegression(max_iter=1000, random_state=42)
    model1.fit(X_train, y_train)
   

    print("✅ Przygotowanie danych zakończone. Dane gotowe do trenowania modeli.")
    
    # Ocena modelu 1
    print("📊 Wyniki - Model 1: Logistic Regression")
    evaluate_model(model1, X_test, y_test)

if __name__ == '__main__':
    main()