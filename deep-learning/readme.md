- Install Python
    ```sh
    python3 -m venv --system-site-packages ./venv
    source ./venv/bin/activate

    pip3 install numpy
    #install keras https://keras.io/about/
    pip3 install tensorflow
    pip3 install keras --upgrade
    #python3 -c 'import keras; print(keras.__version__)'

    # don't exit until you're done using TensorFlow
    deactivate
    ```