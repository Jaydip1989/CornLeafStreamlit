import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras.preprocessing.image import img_to_array

def get_data():
    IMAGE_SIZE = 299
    BATCH_SIZE = 32
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory = "dataset/train/",
        seed = 12,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory='dataset/test/',
        seed = 12,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    classes = train_data.class_names
    return train_data, test_data, classes,IMAGE_SIZE

def get_preprocessed_data(train_data, test_data, IMAGE_SIZE):
    rescale_resize = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.25),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.25)
    ])
    train_data = train_data.map(
        lambda x, y:(rescale_resize(x, training=True), y)
    ).prefetch(buffer_size = tf.data.AUTOTUNE)

    train_ds = train_data.map(
        lambda x, y:(data_augmentation(x, training=True), y)
    ).prefetch(buffer_size = tf.data.AUTOTUNE)

    test_data = test_data.map(
        lambda x, y:(rescale_resize(x, training=False), y)
    ).prefetch(buffer_size = tf.data.AUTOTUNE)
    
    test_ds = test_data.map(
        lambda x, y:(data_augmentation(x, training=False), y)
    ).prefetch(buffer_size = tf.data.AUTOTUNE)
    return train_ds, test_ds

def create_model(IMAGE_SIZE, train_ds, test_ds, classes):
    net = keras.applications.inception_v3.InceptionV3(
        weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    for layer in net.layers:
        layer.trainable = False
    
    model = models.Sequential([
        net,
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(classes), activation="softmax")
    ])
    model.summary()

    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer = optimizers.Adam(learning_rate=0.001),
        metrics = ['acc']
    )
    print("******************************************************************************************")
    print("")
    print("Training")
    print("******************************************************************************************")
    print("")
    history = model.fit(
        train_ds,
        epochs = 10,
        validation_data=test_ds
    )
    print("******************************************************************************************")
    print("")
    print("Evaluating")
    print("******************************************************************************************")
    print("")
    scores = model.evaluate(test_ds, verbose = 1)
    print(f"Loss : {scores[0]} Accuracy:{scores[1]}")
    print("******************************************************************************************")
    print("Testing")
    print("******************************************************************************************")
    print("")
    def predict(model, img):
        img_array = img_to_array(images[i].numpy())
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)

        predicted_class = classes[np.argmax(predictions[0])]
        confidence = round(100*(np.max(predictions[0])), 2)
        return predicted_class, confidence
    
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i+1)
            plt.imshow(images[i].numpy().astype('uint8'))

            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = classes[labels[i]]

            plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}, \n Confidence: {confidence}%")
            plt.axis('off')
        plt.show()
    return model
def main():
    train_data, test_data, classes, IMAGE_SIZE = get_data()
    train_ds, test_ds = get_preprocessed_data(train_data, test_data,IMAGE_SIZE)
    model = create_model(IMAGE_SIZE, train_ds, test_ds,classes)
    model.save("model/CornLeafInception.h5")

if __name__ == "__main__":
    main()
