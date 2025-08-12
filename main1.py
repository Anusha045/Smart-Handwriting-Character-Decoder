import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load EMNIST Letters dataset
print("Starting Handwriting Decoder for Alphabets (Accuracy Focused)...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Normalize images & shift labels from 1-26 to 0-25
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label - 1

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache().shuffle(1000).batch(128).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(ds_train, epochs=5, validation_data=ds_test)

# Evaluate
loss, acc = model.evaluate(ds_test)
print(f"\n✅ Final Accuracy on Test Data: {acc * 100:.2f}%")

# Predict 5 random samples
import numpy as np
letters = "abcdefghijklmnopqrstuvwxyz"
for images, labels in ds_test.take(1):
    preds = model.predict(images[:5])
    for i in range(5):
        true_label = letters[labels[i].numpy()]
        pred_label = letters[np.argmax(preds[i])]
        confidence = np.max(preds[i]) * 100
        print(f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2f}%")
        plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.show()
