from sklearn.model_selection import train_test_split

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X, y, x_train, x_test, y_train, y_test