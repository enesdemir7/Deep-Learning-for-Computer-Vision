import tensorflow as tf
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization , Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError

#Veri Hazırlanması
data = pd.read_csv(r"C:\Users\marlo7\Desktop\DeepLearning for Computer Vision\Building Neural Networks with TensorFlow [Car Price Prediction]\train.csv")
print(data.head())

#sns.pairplot(data[['v.id',"on road old","on road now","years","km","rating","condition","economy","top speed","hp","torque","current price"]], diag_kind='kde')

#sns.pairplot(data[["years","km","rating","condition","economy","top speed","hp","torque","current price"]], diag_kind='kde')
#plt.show()

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
print(tensor_data)

X = tensor_data[:, 3:-1]
print(X[:5])





