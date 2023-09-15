
import mlrun
import pandas as pd
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn import ensemble
from sklearn.model_selection import train_test_split


@mlrun.handler()
def train_model(context, dataset: pd.DataFrame, label_column: str):
    # Initialize our dataframes
    X = dataset.drop(label_column, axis=1)
    y = dataset[label_column]

    # Train/Test split Iris data-set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier()

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    #######################################################################
    apply_mlrun(model, model_name="my_model", x_test=X_test, y_test=y_test)
    #######################################################################

    # Train our model
    model.fit(X_train, y_train)
