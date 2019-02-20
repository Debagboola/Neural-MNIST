#import the important modules
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image

#load the dataset
(train_x, train_y) , (test_x, test_y) = mnist.load_data()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)



#exit(0)

#flatten the image
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)

#convert all your labels to one hot encoded  data
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y,10)


#define your neural network
model = Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(784,)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

#ask keras to compile your model with the right components. then we specify  the matrics here. I am doing this so that my model can report my accuracy to me
model.compile(optimizer=SGD(0.001), loss='categorical_crossentropy', metrics=['accuracy'])



#feed in the training images and their labels. specify a batch size
#model.fit(train_x,train_y, batch_size=32, epochs=10, verbose=1)

#save your model using this code

#model.save('mnist-model.h5')

model.load_weights('mnist-model.h5')

img = test_x[130]
test_img = img.reshape((1,784))

img_class = model.predict_classes(test_img)


classname = img_class[0]

print('Class: ', classname)



#once you have trained  your model, you will have to test the model for accuracy. so use this code:
accuracy = model.evaluate(x=test_x, y=test_y, batch_size = 32)
#print('Accuracy: ', accuracy[1])

img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()