import base64
from io import BytesIO
import numpy
import tensorflow as tf
from PIL import Image
from django.http import HttpResponse
from django.shortcuts import render

model = tf.keras.models.load_model('model.h5')


def image_classifier(request):
    if request.method == "POST":
        data_uri = request.POST['imgBase64']
        num_array = convert_base64_to_numpy_array(data_uri)  # converts base64 image to numpy image
        num_array = numpy.expand_dims(num_array, axis=-1)
        prediction = model.predict(numpy.array([num_array]))
        prediction = numpy.argmax(prediction)
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
