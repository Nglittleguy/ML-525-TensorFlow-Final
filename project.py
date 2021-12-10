import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import datasets, layers, models


# Reading Dataset & Preprocessing
path = "C:/Users/steph/Desktop/ML525/Final/dataset"

img_list = []
class_list = []

class_names = np.array(["cloudy", "rain", "shine", "sunrise"])

for filename in os.listdir(path):
	if filename.startswith("cloudy"):
		classification = 0

	elif filename.startswith("rain"):
		classification = 1

	elif filename.startswith("shine"):
		classification = 2

	elif filename.startswith("sunrise"):
		classification = 3

	img = cv2.imread(os.path.join(path, filename))


	# Catch is error resizing image
	try:
		img_resize = cv2.resize(img, (100,100))
		img_resize = img_resize/255
	except cv2.error:
		print("Failed to resize - skip: ",filename);
		continue;

	# Add to the list, with it's classification
	img_list.append(img_resize)
	class_list.append(classification)


# img_list is (1123, 100, 100, 3), class_list is (1123, 1)

img_train = []
class_train = np.array();
img_test = []
class_test = np.array();

permuted_idx = np.random.permutation(1123)
for pIndex in permuted_idx[0:899]:
	img_train.append(img_list[pIndex])
	class_train.append(class_list[pIndex])
for pIndex in permuted_idx[900:]:
	img_test.append(img_list[pIndex])
	class_test.append(class_list[pIndex])

# img_train(test), class_train(test) is randomized subsets of data

model = models.Sequential()
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(img_train, class_train, epochs=10, validation_data=(img_test, class_test))
model.summary()
model.save('firstModel')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(img_test, class_test, verbose=2)
print(test_acc)




