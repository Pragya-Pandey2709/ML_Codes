#Keras Tuner- Decide Number of Hidden Layers And Neuron In Neural Network
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
df=pd.read_csv('Real_Combine.csv')
df.head()
T	TM	Tm	SLP	H	VV	V	VM	PM 2.5
0	7.4	9.8	4.8	1017.6	93.0	0.5	4.3	9.4	219.720833
1	7.8	12.7	4.4	1018.5	87.0	0.6	4.4	11.1	182.187500
2	6.7	13.4	2.4	1019.4	82.0	0.6	4.8	11.1	154.037500
3	8.6	15.5	3.3	1018.7	72.0	0.8	8.1	20.6	223.208333
4	12.4	20.9	4.4	1017.3	61.0	1.3	8.7	22.2	200.645833
X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features
Hyperparameters
How many number of hidden layers we should have?
How many number of neurons we should have in hidden layers?
Learning Rate
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model
tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Air Quality Index')
tuner.search_space_summary()
Search space summary
|-Default search space size: 4
num_layers (Int)
|-default: None
|-max_value: 20
|-min_value: 2
|-sampling: None
|-step: 1
units_0 (Int)
|-default: None
|-max_value: 512
|-min_value: 32
|-sampling: None
|-step: 32
units_1 (Int)
|-default: None
|-max_value: 512
|-min_value: 32
|-sampling: None
|-step: 32
learning_rate (Choice)
|-default: 0.01
|-ordered: True
|-values: [0.01, 0.001, 0.0001]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 116.8132 - mean_absolute_error: 116.813 - ETA: 0s - loss: 76.3447 - mean_absolute_error: 76.3447  - 0s 18ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.4238 - val_mean_absolute_error: 63.4238
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 72.3939 - mean_absolute_error: 72.393 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 16ms/step - loss: nan - mean_absolute_error: nan - val_loss: 62.7431 - val_mean_absolute_error: 62.7431
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 78.7923 - mean_absolute_error: 78.792 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 19ms/step - loss: nan - mean_absolute_error: nan - val_loss: 47.5451 - val_mean_absolute_error: 47.5451
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 48.1353 - mean_absolute_error: 48.135 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 72.8017 - val_mean_absolute_error: 72.8017
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 58.7824 - mean_absolute_error: 58.782 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 55.2045 - val_mean_absolute_error: 55.2045
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 127.8929 - mean_absolute_error: 127.892 - 0s 6ms/step - loss: nan - mean_absolute_error: nan - val_loss: 71.2169 - val_mean_absolute_error: 71.2169
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 81.4796 - mean_absolute_error: 81.479 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 59.2815 - val_mean_absolute_error: 59.2815
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 65.3511 - mean_absolute_error: 65.351 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 47.8693 - val_mean_absolute_error: 47.8693
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 26.0163 - mean_absolute_error: 26.016 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 57.7024 - val_mean_absolute_error: 57.7024
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 48.9063 - mean_absolute_error: 48.906 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 49.2807 - val_mean_absolute_error: 49.2807
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 132.7942 - mean_absolute_error: 132.794 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 0s 7ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.7852 - val_mean_absolute_error: 66.7852
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 91.3190 - mean_absolute_error: 91.319 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 65.7265 - val_mean_absolute_error: 65.7265
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 65.8415 - mean_absolute_error: 65.841 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 60.6193 - val_mean_absolute_error: 60.6193
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 64.0055 - mean_absolute_error: 64.005 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 47.9010 - val_mean_absolute_error: 47.9010
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 53.0121 - mean_absolute_error: 53.012 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 51.7172 - val_mean_absolute_error: 51.7172
C:\Users\win10\anaconda3\envs\myenv\lib\site-packages\kerastuner\engine\metrics_tracking.py:92: RuntimeWarning: All-NaN axis encountered
  return np.nanmin(values)
Trial complete
Trial summary
|-Trial ID: bd9d67176a48257acd476a792a58666c
|-Score: 47.77180608113607
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.01
|-num_layers: 10
|-units_0: 192
|-units_1: 256
|-units_2: 32
|-units_3: 32
|-units_4: 32
|-units_5: 32
|-units_6: 32
|-units_7: 32
|-units_8: 32
|-units_9: 32
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 115.1197 - mean_absolute_error: 115.119 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 1s 29ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.2618 - val_mean_absolute_error: 64.2618
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 70.6723 - mean_absolute_error: 70.672 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 1s 27ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.4300 - val_mean_absolute_error: 63.4300
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 50.7973 - mean_absolute_error: 50.797 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.6015 - val_mean_absolute_error: 66.6015
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 84.8132 - mean_absolute_error: 84.813 - ETA: 0s - loss: 70.3534 - mean_absolute_error: 70.353 - 1s 27ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.2408 - val_mean_absolute_error: 63.2408
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 52.6199 - mean_absolute_error: 52.619 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 69.6594 - val_mean_absolute_error: 69.6594
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 130.5468 - mean_absolute_error: 130.546 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 0s 8ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.0518 - val_mean_absolute_error: 64.0518
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 82.7737 - mean_absolute_error: 82.773 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.5639 - val_mean_absolute_error: 64.5639
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 55.6748 - mean_absolute_error: 55.674 - ETA: 0s - loss: 71.1010 - mean_absolute_error: 71.101 - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5915 - val_mean_absolute_error: 63.5915
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 69.2862 - mean_absolute_error: 69.286 - ETA: 0s - loss: 67.9955 - mean_absolute_error: 67.995 - 0s 5ms/step - loss: nan - mean_absolute_error: nan - val_loss: 65.5058 - val_mean_absolute_error: 65.5058
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 64.8581 - mean_absolute_error: 64.858 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 5ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.2052 - val_mean_absolute_error: 66.2052
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 93.9155 - mean_absolute_error: 93.915 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 8ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.8856 - val_mean_absolute_error: 63.8856
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: nan - mean_absolute_error: na - ETA: 0s - loss: nan - mean_absolute_error: na - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.4444 - val_mean_absolute_error: 64.4444
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 56.2157 - mean_absolute_error: 56.215 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.0128 - val_mean_absolute_error: 64.0128
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 77.8994 - mean_absolute_error: 77.899 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 72.6649 - val_mean_absolute_error: 72.6649
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 65.4756 - mean_absolute_error: 65.475 - ETA: 0s - loss: 67.4593 - mean_absolute_error: 67.459 - 1s 27ms/step - loss: nan - mean_absolute_error: nan - val_loss: 62.9103 - val_mean_absolute_error: 62.9103
C:\Users\win10\anaconda3\envs\myenv\lib\site-packages\kerastuner\engine\metrics_tracking.py:92: RuntimeWarning: All-NaN axis encountered
  return np.nanmin(values)
Trial complete
Trial summary
|-Trial ID: 41a673bc36d9b7da8bc20594dd299cff
|-Score: 63.24752426147461
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.001
|-num_layers: 15
|-units_0: 160
|-units_1: 64
|-units_10: 32
|-units_11: 32
|-units_12: 32
|-units_13: 32
|-units_14: 32
|-units_2: 384
|-units_3: 480
|-units_4: 384
|-units_5: 352
|-units_6: 192
|-units_7: 384
|-units_8: 288
|-units_9: 64
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 116.3867 - mean_absolute_error: 116.386 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 1s 24ms/step - loss: nan - mean_absolute_error: nan - val_loss: 68.6886 - val_mean_absolute_error: 68.6886
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 83.5364 - mean_absolute_error: 83.536 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 1s 23ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.6457 - val_mean_absolute_error: 63.6457
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 68.2929 - mean_absolute_error: 68.292 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 65.1698 - val_mean_absolute_error: 65.1698
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 56.6373 - mean_absolute_error: 56.637 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.3594 - val_mean_absolute_error: 64.3594
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 61.8076 - mean_absolute_error: 61.807 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 20ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5364 - val_mean_absolute_error: 63.5364
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 118.2550 - mean_absolute_error: 118.255 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 0s 7ms/step - loss: nan - mean_absolute_error: nan - val_loss: 65.2420 - val_mean_absolute_error: 65.2420
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 69.8192 - mean_absolute_error: 69.819 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.6967 - val_mean_absolute_error: 63.6967
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 52.9929 - mean_absolute_error: 52.992 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.0549 - val_mean_absolute_error: 66.0549
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 65.4321 - mean_absolute_error: 65.432 - ETA: 0s - loss: 66.1418 - mean_absolute_error: 66.141 - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.8149 - val_mean_absolute_error: 64.8149
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 70.4234 - mean_absolute_error: 70.423 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.2624 - val_mean_absolute_error: 64.2624
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 71.8871 - mean_absolute_error: 71.887 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 7ms/step - loss: nan - mean_absolute_error: nan - val_loss: 69.0089 - val_mean_absolute_error: 69.0089
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 56.7036 - mean_absolute_error: 56.703 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.8348 - val_mean_absolute_error: 63.8348
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 40.0668 - mean_absolute_error: 40.066 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.8099 - val_mean_absolute_error: 64.8099
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 53.5800 - mean_absolute_error: 53.580 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.7348 - val_mean_absolute_error: 63.7348
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 89.3396 - mean_absolute_error: 89.339 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.7883 - val_mean_absolute_error: 64.7883
C:\Users\win10\anaconda3\envs\myenv\lib\site-packages\kerastuner\engine\metrics_tracking.py:92: RuntimeWarning: All-NaN axis encountered
  return np.nanmin(values)
Trial complete
Trial summary
|-Trial ID: d5a4e9bfb2ddc879dee8ded6996324a9
|-Score: 63.65599568684896
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.0001
|-num_layers: 11
|-units_0: 288
|-units_1: 160
|-units_10: 128
|-units_11: 64
|-units_12: 448
|-units_13: 384
|-units_14: 384
|-units_2: 64
|-units_3: 320
|-units_4: 288
|-units_5: 320
|-units_6: 96
|-units_7: 96
|-units_8: 352
|-units_9: 192
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 178.2827 - mean_absolute_error: 178.282 - 1s 22ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.0686 - val_mean_absolute_error: 64.0686
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 55.9803 - mean_absolute_error: 55.980 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 65.5033 - val_mean_absolute_error: 65.5033
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 65.5051 - mean_absolute_error: 65.505 - 0s 20ms/step - loss: nan - mean_absolute_error: nan - val_loss: 61.1847 - val_mean_absolute_error: 61.1847
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 63.5379 - mean_absolute_error: 63.537 - 0s 18ms/step - loss: nan - mean_absolute_error: nan - val_loss: 59.7015 - val_mean_absolute_error: 59.7015
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 57.2574 - mean_absolute_error: 57.257 - 0s 19ms/step - loss: nan - mean_absolute_error: nan - val_loss: 56.0679 - val_mean_absolute_error: 56.0679
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 98.6499 - mean_absolute_error: 98.649 - 0s 5ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.8493 - val_mean_absolute_error: 66.8493
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 71.4547 - mean_absolute_error: 71.454 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.0680 - val_mean_absolute_error: 63.0680
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 68.1744 - mean_absolute_error: 68.174 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 61.9647 - val_mean_absolute_error: 61.9647
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 67.6296 - mean_absolute_error: 67.629 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.9284 - val_mean_absolute_error: 64.9284
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 64.9716 - mean_absolute_error: 64.971 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.2758 - val_mean_absolute_error: 64.2758
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 115.2061 - mean_absolute_error: 115.206 - 0s 5ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.7263 - val_mean_absolute_error: 63.7263
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 76.3914 - mean_absolute_error: 76.391 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5010 - val_mean_absolute_error: 63.5010
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 56.3952 - mean_absolute_error: 56.395 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 72.2587 - val_mean_absolute_error: 72.2587
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 67.4230 - mean_absolute_error: 67.423 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 58.8266 - val_mean_absolute_error: 58.8266
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 49.3929 - mean_absolute_error: 49.392 - 0s 3ms/step - loss: nan - mean_absolute_error: nan - val_loss: 61.4033 - val_mean_absolute_error: 61.4033
C:\Users\win10\anaconda3\envs\myenv\lib\site-packages\kerastuner\engine\metrics_tracking.py:92: RuntimeWarning: All-NaN axis encountered
  return np.nanmin(values)
Trial complete
Trial summary
|-Trial ID: 94639d964caf10b4baa7059e8d927d02
|-Score: 58.95306905110677
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.001
|-num_layers: 3
|-units_0: 480
|-units_1: 480
|-units_10: 352
|-units_11: 416
|-units_12: 384
|-units_13: 320
|-units_14: 160
|-units_2: 512
|-units_3: 64
|-units_4: 448
|-units_5: 128
|-units_6: 416
|-units_7: 64
|-units_8: 352
|-units_9: 192
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 81.2644 - mean_absolute_error: 81.264 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 1s 27ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.7037 - val_mean_absolute_error: 64.7037
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 83.1217 - mean_absolute_error: 83.121 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 1s 23ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5881 - val_mean_absolute_error: 63.5881
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 71.5101 - mean_absolute_error: 71.510 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.8098 - val_mean_absolute_error: 64.8098
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: nan - mean_absolute_error: na - ETA: 0s - loss: nan - mean_absolute_error: na - 1s 24ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5553 - val_mean_absolute_error: 63.5553
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 66.2654 - mean_absolute_error: 66.265 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.5349 - val_mean_absolute_error: 64.5349
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 130.7908 - mean_absolute_error: 130.790 - ETA: 0s - loss: nan - mean_absolute_error: nan          - 0s 7ms/step - loss: nan - mean_absolute_error: nan - val_loss: 71.2532 - val_mean_absolute_error: 71.2532
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: nan - mean_absolute_error: na - ETA: 0s - loss: nan - mean_absolute_error: na - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.0055 - val_mean_absolute_error: 64.0055
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 70.5603 - mean_absolute_error: 70.560 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.7454 - val_mean_absolute_error: 63.7454
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: 59.7425 - mean_absolute_error: 59.742 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.2013 - val_mean_absolute_error: 64.2013
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 64.5670 - mean_absolute_error: 64.567 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.8017 - val_mean_absolute_error: 63.8017
Epoch 1/5
WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

24/24 [==============================] - ETA: 0s - loss: 101.1746 - mean_absolute_error: 101.174 - ETA: 0s - loss: 98.0866 - mean_absolute_error: 98.0866  - 0s 7ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.6381 - val_mean_absolute_error: 63.6381
Epoch 2/5
24/24 [==============================] - ETA: 0s - loss: 68.8109 - mean_absolute_error: 68.810 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 16ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.5001 - val_mean_absolute_error: 63.5001
Epoch 3/5
24/24 [==============================] - ETA: 0s - loss: 49.7558 - mean_absolute_error: 49.755 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 1s 22ms/step - loss: nan - mean_absolute_error: nan - val_loss: 63.4895 - val_mean_absolute_error: 63.4895
Epoch 4/5
24/24 [==============================] - ETA: 0s - loss: nan - mean_absolute_error: na - ETA: 0s - loss: nan - mean_absolute_error: na - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 64.1716 - val_mean_absolute_error: 64.1716
Epoch 5/5
24/24 [==============================] - ETA: 0s - loss: 56.0330 - mean_absolute_error: 56.033 - ETA: 0s - loss: nan - mean_absolute_error: nan        - 0s 4ms/step - loss: nan - mean_absolute_error: nan - val_loss: 66.8828 - val_mean_absolute_error: 66.8828
C:\Users\win10\anaconda3\envs\myenv\lib\site-packages\kerastuner\engine\metrics_tracking.py:92: RuntimeWarning: All-NaN axis encountered
  return np.nanmin(values)
Trial complete
Trial summary
|-Trial ID: 421f2a1f272dd408cdfecc7b2f9c244f
|-Score: 63.596744537353516
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.0001
|-num_layers: 11
|-units_0: 224
|-units_1: 64
|-units_10: 160
|-units_11: 256
|-units_12: 480
|-units_13: 256
|-units_14: 64
|-units_2: 512
|-units_3: 416
|-units_4: 480
|-units_5: 192
|-units_6: 256
|-units_7: 224
|-units_8: 64
|-units_9: 256
INFO:tensorflow:Oracle triggered exit
tuner.results_summary()
Results summary
|-Results in project\Air Quality Index
|-Showing 10 best trials
|-Objective(name='val_mean_absolute_error', direction='min')
Trial summary
|-Trial ID: bd9d67176a48257acd476a792a58666c
|-Score: 47.77180608113607
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.01
|-num_layers: 10
|-units_0: 192
|-units_1: 256
|-units_2: 32
|-units_3: 32
|-units_4: 32
|-units_5: 32
|-units_6: 32
|-units_7: 32
|-units_8: 32
|-units_9: 32
Trial summary
|-Trial ID: 94639d964caf10b4baa7059e8d927d02
|-Score: 58.95306905110677
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.001
|-num_layers: 3
|-units_0: 480
|-units_1: 480
|-units_10: 352
|-units_11: 416
|-units_12: 384
|-units_13: 320
|-units_14: 160
|-units_2: 512
|-units_3: 64
|-units_4: 448
|-units_5: 128
|-units_6: 416
|-units_7: 64
|-units_8: 352
|-units_9: 192
Trial summary
|-Trial ID: 41a673bc36d9b7da8bc20594dd299cff
|-Score: 63.24752426147461
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.001
|-num_layers: 15
|-units_0: 160
|-units_1: 64
|-units_10: 32
|-units_11: 32
|-units_12: 32
|-units_13: 32
|-units_14: 32
|-units_2: 384
|-units_3: 480
|-units_4: 384
|-units_5: 352
|-units_6: 192
|-units_7: 384
|-units_8: 288
|-units_9: 64
Trial summary
|-Trial ID: 421f2a1f272dd408cdfecc7b2f9c244f
|-Score: 63.596744537353516
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.0001
|-num_layers: 11
|-units_0: 224
|-units_1: 64
|-units_10: 160
|-units_11: 256
|-units_12: 480
|-units_13: 256
|-units_14: 64
|-units_2: 512
|-units_3: 416
|-units_4: 480
|-units_5: 192
|-units_6: 256
|-units_7: 224
|-units_8: 64
|-units_9: 256
Trial summary
|-Trial ID: d5a4e9bfb2ddc879dee8ded6996324a9
|-Score: 63.65599568684896
|-Best step: 0
Hyperparameters:
|-learning_rate: 0.0001
|-num_layers: 11
|-units_0: 288
|-units_1: 160
|-units_10: 128
|-units_11: 64
|-units_12: 448
|-units_13: 384
|-units_14: 384
|-units_2: 64
|-units_3: 320
|-units_4: 288
|-units_5: 320
|-units_6: 96
|-units_7: 96
|-units_8: 352
|-units_9: 192
 
