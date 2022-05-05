from openvino.runtime import Core
import cv2
import numpy as np

ie = Core()

model = ie.read_model(model='models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml')
compiled_model = ie.compile_model(model=model, device_name='CPU')

input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

print(input_layer, output_layer)

image_filename = 'image.jpg'
image = cv2.imread(image_filename)
print('image shape:', image.shape)

# N,C,H,W = batch size, number of channels, height, width
N, C, H, W = input_layer.shape
# OpenCV resize expects the destination size as (width, height)
resized_image = cv2.resize(src=image, dsize=(W, H))
print('resized image shape: ', resized_image.shape)

input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
input_data.shape

result = compiled_model([input_data])

print(result)