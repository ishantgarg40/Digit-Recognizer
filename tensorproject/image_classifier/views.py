import base64
from io import BytesIO

import numpy
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
from PIL import Image
from django.http import HttpResponse
from django.shortcuts import render

print("HERE TENSORFLOW BEGINS....................")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = numpy.expand_dims(x_train, axis=-1), numpy.expand_dims(x_test, axis=-1)

inputs = kl.Input(shape=(28, 28, 1))
c = kl.Conv2D(32, (3, 3), padding="valid", activation=tf.nn.relu)(inputs)
m = kl.MaxPool2D((2, 2), (2, 2))(c)
c = kl.Conv2D(64, (3, 3), padding="valid", activation=tf.nn.relu)(m)
m = kl.MaxPool2D((2, 2), (2, 2))(c)
f = kl.Flatten()(m)
outputs = kl.Dense(10, activation=tf.nn.softmax)(f)

model = km.Model(inputs, outputs)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
model.fit(x_train, y_train, epochs=5)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss} and Test Acc: {acc}")

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>XXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def image_classifier(request):
    print("REQUEST CAME>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL", request.method)
    if request.method == "POST":
        data_uri = request.POST['imgBase64']
        num_array = convert_base64_to_numpy_array(data_uri)  # converts base64 image to numpy image
        num_array = numpy.expand_dims(num_array, axis=-1)
        prediction = model.predict(numpy.array([num_array]))
        prediction = numpy.argmax(prediction)
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTttt", prediction)
        return HttpResponse(prediction)

    return render(request, 'image_classifier/index.html')


def convert_base64_to_numpy_array(data_uri):
    dimensions = (28, 28)
    encoded_image = data_uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image))
    img = img.resize(dimensions, Image.ANTIALIAS)
    pixels = numpy.array(img)[:, :, 0]
    return pixels/255.0
