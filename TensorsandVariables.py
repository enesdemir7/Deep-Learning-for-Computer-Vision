import tensorflow as tf
import numpy as np

'''
tensor_zero_d = tf.constant(4)
print(tensor_zero_d)

tensor_one_d = tf.constant([2,0,-3,90.], dtype=tf.float32)
print(tensor_one_d)

tensor_two_d = tf.constant([[1,2,0],
                           [3,6,3],
                           [6,7,9]])

print(tensor_two_d)

#casting işlemleri
casted_tensor_one_d=tf.cast(tensor_one_d, dtype=tf.int16)
print(tensor_one_d)
print(casted_tensor_one_d)

np_array = np.array([1,2,4])
print(np_array)

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)



eye_tensor = tf.eye(
    num_rows=5,
    num_columns=None,
    batch_shape=[2],
    dtype=tf.dtypes.float32,
    name=None 
)

print(eye_tensor)



fill_tensor = tf.fill(
    [1,2,3,4],5, name=None
)

print(fill_tensor)

random_tensor = tf.random.uniform(
    [5,],
    minval=0,
    maxval=100,
    dtype=tf.dtypes.int32,
    seed=None,
    name=None
)

print(random_tensor)


tensor_indexed = tf.constant([3,6,2,4,6,66,7])
print(tensor_indexed)
print(tensor_indexed[0:4])
print(tensor_indexed[:4])
print(tensor_indexed[3:])
print(tensor_indexed[3:-1])


tensor_two_d = tf.constant(
    [
        [1,2,9],
        [3,41,1],
        [2,1,3],
        [8,8,7]
    ]
)
print(tensor_two_d[:,:2 ])
 '''

#math operation
x = tf.constant([-2.25, 3.25])
print(tf.abs(x))
x_complex = tf.constant([-2.25 + 3j])
print(tf.abs(x_complex))

