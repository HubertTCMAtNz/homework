import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

x_train = np.array([[1, 1], [1, 2], [1, 3], [1, 4],
    [2, 1], [2, 2], [2, 3], [2, 4],
    [3, 1], [3, 2], [3, 3], [3, 4],
    [4, 1], [4, 2], [4, 3]])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

print("The input vector:\n", x_train.shape)
print("The input vector:\n", x_train)

y_train = np.array([5, 3, 8, 7,
    12, 15, 49, 56,
    3, 9, 4, 12,
    18, 27, 36])

batch_size = 1
epochs = 15
def model_one():
    input_shape = (2, 1,)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #layers.Conv1D(64, 1, activation="relu", input_shape=(2, 1)),
            #layers.Dropout(0.5),
            #layers.Dense(3, activation="softmax"),
            #layers.Dense(3, activation="softmax"),
            layers.Dense(1, activation="softmax"),

            #layers.Dense(16, activation="relu"),
            #layers.MaxPooling1D(),
            #layers.Flatten(),
            #layers.Dense(3, activation = 'softmax')
        ]
    )
    #model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

def main():
    model_one()
    print("Hello World!")

if __name__ == "__main__":
    main()