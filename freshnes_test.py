import cv2
import tensorflow as tf
from keras.models import load_model
import os

IMG_PATH = 'test imeges/apple.jpg'

model = load_model('fruits_model_final.h5')
img = cv2.imread(IMG_PATH)
print(img.shape)
img = cv2.resize(img,(128,128))
img = img / 255
img = tf.expand_dims(img,axis=0)

prediction = model.predict(img)

DATASET_PATH ='Fruit Freshness Dataset/Apple'
classes = os.listdir(DATASET_PATH)
class_num = prediction.argmax()
class_name = classes[class_num]
print(f'predicted class: {class_name}')
