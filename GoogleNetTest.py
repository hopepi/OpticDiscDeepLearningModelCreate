import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from pathlib import Path
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten
from keras.src.optimizers import Adam
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.utils.image_utils import img_to_array,load_img
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


def create_dataframe(directory):
    filepaths = []
    labels = []

    for label_dir in directory.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for img_file in label_dir.iterdir():
                file_extension = img_file.suffix
                if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
                    filepaths.append(str(img_file))
                    labels.append(label)

    if not filepaths or not labels:
        raise ValueError("Filepaths or labels are empty.")
    if len(filepaths) != len(labels):
        raise ValueError("Filepaths and labels must have the same length.")

    data = {'Filepath': filepaths, 'Label': labels}
    df = pd.DataFrame(data)
    return df

#base_dir = Path('C:\\Users\\umutk\\OneDrive\\Belgeler\\Glaucoma_Sharpened_COLOR_CLAHE') En optimal %93
#base_dir = Path('C:\\Users\\umutk\\OneDrive\\Belgeler\\Fundus_Processed') %73
base_dir = Path('C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset')
train_dir = base_dir / "train"
test_dir = base_dir / "test"
df_test = create_dataframe(test_dir)
df_train = create_dataframe(train_dir)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='Filepath',
    y_col='Label',
    color_mode='rgb',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16,
    shuffle=True,
    seed=42,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='Filepath',
    y_col='Label',
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=16,
    shuffle=False
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
# Önceden eğitilmiş katmanları dondurmayınca %92 ama aşırı öğrenme olmaması adına False yapıyorum

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1028, activation='relu'))
model.add(Dropout(0.2))


num_classes = len(df_train['Label'].unique())
model.add(Dense(num_classes, activation='softmax'))

optimizer=Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=40,
    validation_data=test_generator,
    validation_steps=len(test_generator),
)

loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

def manual_test(filepaths, model, target_size=(224, 224)):
    for filepath in filepaths:
        img = load_img(filepath, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0


        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        class_name = list(test_generator.class_indices.keys())[predicted_class]
        print(f"{filepath}: Tahmin edilen sınıf - {class_name}")


hasta_files = [
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Glaucoma\\225_COLORCLAHE_SHARPENED_PROCESSED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Glaucoma\\617_COLORCLAHE_SHARPENED_PROCESSED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Glaucoma\\613_COLORCLAHE_SHARPENED_PROCESSED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\test\\Glaucoma\\Im326_g_ACRIMA_COLORCLAHE_SHARPENED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\test\\Glaucoma\\Im483_g_ACRIMA_COLORCLAHE_SHARPENED.png",
]
hasta_degil_files = [
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Non Glaucoma\\022_COLORCLAHE_SHARPENED_PROCESSED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Non Glaucoma\\035_COLORCLAHE_SHARPENED_PROCESSED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\train\\Non Glaucoma\\Im084_ACRIMA_COLORCLAHE_SHARPENED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\test\\Non Glaucoma\\Im018_ACRIMA_COLORCLAHE_SHARPENED.png",
    "C:\\Users\\umutk\\OneDrive\\Belgeler\\NewDataset\\test\\Non Glaucoma\\052_COLORCLAHE_SHARPENED_PROCESSED.png",
]

# Testi çalıştırın
print("Hasta Testleri:")
manual_test(hasta_files, model)

print("\nHasta Değil Testleri:")
manual_test(hasta_degil_files, model)


# Test veri setindeki gerçek etiketleri alma
test_labels = test_generator.classes

# Modelin tahminlerini alma
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Confusion matrix oluşturma
cm = confusion_matrix(test_labels, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Sınıflar için rapor oluşturma
report = classification_report(test_labels, predicted_classes, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(report)



"""
%86~~%90

Test Accuracy: 0.8626315689086914

Confusion Matrix:
[[67 17]
 [11 95]]
 

Classification Report:
              precision    recall  f1-score   support

    Glaucoma       0.86      0.80      0.83        84
Non Glaucoma       0.85      0.90      0.87       106

    accuracy                           0.85       190
   macro avg       0.85      0.85      0.85       190
weighted avg       0.85      0.85      0.85       190
"""