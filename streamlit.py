import numpy as np
import pickle
import tensorflow as tf
import streamlit as st
from tensorflow import keras




# Read dictionary pkl file
with open('C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\2nd Sem\\CMSE 890 applied ML\\final submittion\\CMSE890\\kaggle\\word_2_indices.pkl', 'rb') as fp:
    word_2_indices = pickle.load(fp)
print(word_2_indices)

with open('C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\2nd Sem\\CMSE 890 applied ML\\final submittion\\CMSE890\\kaggle\\indices_2_word.pkl', 'rb') as fp:
    indices_2_word = pickle.load(fp)
print(indices_2_word)

embedding_size = 128
max_len = 40

from PIL import Image
import cv2
def preprocessing(img_path):
    im =  tf.keras.utils.load_img("C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\2nd Sem\\CMSE 890 applied ML\\final submittion\\CMSE890\\temp.png",target_size=(224,224,3))
    # print(im)
    im = tf.keras.utils.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = tf.keras.utils.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
     
    return ' '.join(start_word[1:-1])
from PIL import Image
import io
def saveImage(byteImage):
    bytesImg = io.BytesIO(byteImage)
    imgFile = Image.open(bytesImg)   
    imgFile.resize((224,224))
    return imgFile

model = keras.models.load_model('C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\2nd Sem\\CMSE 890 applied ML\\final submittion\\CMSE890\\kaggle\\model_weights_1.h5')

st.write("upload image")

img = st.file_uploader("Choose a img file", accept_multiple_files=False,type =['png', 'jpg'] )
import PIL.Image as Image 
if img:
    print(type(img))
    print(img)
    file = bytearray(img.read())
    image = Image.open(io.BytesIO(file))
    image.save("temp.png")


    image.show()
    print("22222222222222")
    print(image)
    # path = saveImage(file)
    st.image(image)
    resnet = tf.keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    #print(path)
    test_img = get_encoding(resnet, image)
    Argmax_Search = predict_captions(test_img)
    st.write()
    st.write(Argmax_Search)