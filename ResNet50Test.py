import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from pathlib import Path
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten
from keras.src.optimizers import Adam
from keras.src.applications.resnet_v2 import ResNet50V2
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
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
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

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Önceden eğitilmiş katmanları dondur

# Kendi modelinizi oluşturun
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1028, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

num_classes = len(df_train['Label'].unique())
model.add(Dense(num_classes, activation='softmax'))

optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

# Modeli derle
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=40,  # Epoch sayısını artırdık
    validation_data=test_generator,
    validation_steps=len(test_generator),
)

# Değerlendirme
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


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
V1
%60~~%62

Confusion Matrix:
[[40 44]
 [30 76]]

Classification Report:
              precision    recall  f1-score   support

    Glaucoma       0.57      0.48      0.52        84
Non Glaucoma       0.63      0.72      0.67       106

    accuracy                           0.61       190
   macro avg       0.60      0.60      0.60       190
weighted avg       0.61      0.61      0.60       190


V2
%77~~%80

Confusion Matrix:
[[ 45  39]
 [  4 102]]

Classification Report:
              precision    recall  f1-score   support

    Glaucoma       0.92      0.54      0.68        84
Non Glaucoma       0.72      0.96      0.83       106

    accuracy                           0.77       190
   macro avg       0.82      0.75      0.75       190
weighted avg       0.81      0.77      0.76       190
"""