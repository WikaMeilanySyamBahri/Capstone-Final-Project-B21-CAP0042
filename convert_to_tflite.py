import tensorflow as tf
 #import frozen graph (.pb model)
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = 'path/to/frozen_inference__graph.pb', 
    input_arrays = ['Input_Tensor_Name'],
    output_arrays = ['Output_Tensor_Name'] 
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)