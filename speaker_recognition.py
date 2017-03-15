# !/usr/bin/env python
# !/usr/local/bin/python
# !/usr/bin/env PYTHONIOENCODING="utf-8" python
import os

import tflearn
import recorded_data as data

# Simple speaker recognition demo!
# TODO contribute
import tensorflow as tf
print('You are using tensorflow v.' + tf.__version__)
if tf.__version__ >= '0.12' and os.name == 'nt':
    print('sorry, tflearn is not supported in tensorflow 0.12 on windows yet!(?)')
    quit()

speakers = data.get_speakers()
number_classes = len(speakers)
print('speakers', speakers)

batch = data.wave_batch_generator(batch_size=20, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
X, Y = next(batch)


# Classifier
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 16384])  # n wave chunks
net = tflearn.fully_connected(net, 128)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)

input_file = 'Test/sample.wav'  # valeria
demo = data.load_wav_file(data.path + input_file)
result = model.predict([demo])
result = data.one_hot_to_item(result, speakers)
print('predicted speaker for %s : result = %s ' % (input_file, str(result)))
