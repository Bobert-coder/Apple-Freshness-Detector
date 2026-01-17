import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import load_model

DATASET_PATH ='Fruit Freshness Dataset/Apple'
classes = os.listdir(DATASET_PATH)
num_classes = len(classes)
data_generator = ImageDataGenerator(rescale=1/255,validation_split=0.2)

train_data = data_generator.flow_from_directory(
    DATASET_PATH,
    target_size= (128,128),
    class_mode= 'categorical',
    subset='training'

)
test_data = data_generator.flow_from_directory(
    DATASET_PATH,
    target_size= (128,128),
    class_mode= 'categorical',
    subset='validation'
)

model = load_model('fruits_model_final.h5')

loss, accuracy = model.evaluate(test_data)
print(f'accuracy: {accuracy}')