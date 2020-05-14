import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """

    np.save('accuracy.npy', model_history.history['accuracy'])
    np.save('val_accuracy.npy', model_history.history['val_accuracy'])
    np.save('loss.npy', model_history.history['loss'])
    np.save('val_loss.npy', model_history.history['val_loss'])

    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    # axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    # axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot_model.png')
    # plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/dev'

num_train = 46485 #28709
num_val = 7216 #7178
batch_size = 128
num_epoch = 40

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation='softmax'))




# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            shuffle=True,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    model.save('model.h5')
    plot_model_history(model_info)



# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model = load_model('model.h5')
    print(model.summary())

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}


    test_dir = 'data/test'
    test_PATH = os.path.join(os.getcwd(), test_dir)

    actual_label = []
    predicted_label = []

    print('Predicting...')
    for em in os.listdir(test_PATH):
        target = os.path.join(test_PATH, em)
        print(target)
        for file in os.listdir(target):
            img = cv2.imread(os.path.join(target, file), 0)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img, (48, 48)), -1), 0)
            predictions = model.predict(cropped_img)
            maxindex = int(np.argmax(predictions))
            actual_label.append(emotion_dict[em])
            predicted_label.append(maxindex)

    actual_label = np.array(actual_label)
    predicted_label = np.array(predicted_label)

    print(actual_label.shape)
    print(predicted_label.shape)

    print("Confusion Matrix :")
    print(confusion_matrix(actual_label, predicted_label))
    print("Classification Report :")
    print(classification_report(actual_label, predicted_label, digits=4))
    print('Weighted FScore: \n ', precision_recall_fscore_support(actual_label, predicted_label, average='weighted'))

