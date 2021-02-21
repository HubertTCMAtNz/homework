import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def model_one():
    x = [item for item in range(1, 101)]
    y = [2 * item + 5 for item in x]
    x_train = np.array(x).reshape(len(x), 1)
    y_train = np.array(y).reshape(len(y), 1)
    model = keras.Sequential([
        keras.Input(shape=(1)),
        layers.Dense(1, activation="relu"),
    ])
    model.summary()
    batch_size = 4
    epochs = 300
    #model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)
    return model


def main():
    model_one()
    print("Hello World!")


if __name__ == "__main__":
    main()
