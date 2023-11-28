from flask import Flask,render_template,request,send_from_directory

# Required For Model
import tensorflow as tf
import cv2
import numpy as np
import base64
from io import BytesIO



def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize the image to the desired size
    img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

x_pre=preprocess('train_1.png_tile_3_0.png')
x_post=preprocess('train_1.png_tile_3_0_p.png')


app=Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')



@app.route("/cd")
def changedetection():
        return render_template('input.html')
    


@app.route("/cd",methods=["POST"])
def predict():
    try:
        # imagefile=request.files['prefile']
    
        # imagepath="./images/"+"pre.png"
        # print(imagefile.filename)
        # imagefile.save(imagepath)
        
        imagefile=request.files['prefile']
    
        imagepath="./static/"+"pre.png"
        print(imagefile.filename)
        imagefile.save(imagepath)
        return render_template('input.html',preimage='hello')
    except:
        imagefile=request.files['postfile']
        imagepath="./static/"+"post.png"
        print(imagefile.filename)
        imagefile.save(imagepath)

        
        return render_template('input.html',preimage=True,postimage="hello")
    
@app.route("/predict",methods=["POST"])
def predict_image():
    model = tf.keras.models.load_model('final_model.h5',compile=False)
    x_pre=preprocess('static\pre.png')
    x_post=preprocess('static\post.png')
    pred_img = model.predict([x_pre,x_post])
    threshold = 0.5
    pred_img_binary = (pred_img > threshold).astype(np.uint8)

        # Save the processed prediction as an image
    output_path = "./static/predicted_change_map.png"
    cv2.imwrite(output_path, pred_img_binary[0, :, :, 0] * 255)
    return render_template('predict.html', result=True, change_map='predicted_change_map.png')


@app.route("/ld")
def changedetection_ld():
        return render_template('inputld.html')

@app.route("/ld",methods=["POST"])
def predictld():
    try:
        # imagefile=request.files['prefile']
    
        # imagepath="./images/"+"pre.png"
        # print(imagefile.filename)
        # imagefile.save(imagepath)
        
        imagefile=request.files['prefile']
    
        imagepath="./static/"+"pre.png"
        print(imagefile.filename)
        imagefile.save(imagepath)
        return render_template('inputld.html',preimage='hello')
    except:
        imagefile=request.files['postfile']
        imagepath="./static/"+"post.png"
        print(imagefile.filename)
        imagefile.save(imagepath)

        
        return render_template('inputld.html',preimage=True,postimage="hello")
    
@app.route("/predictld",methods=["POST"])
def predict_image_ld():
    model = tf.keras.models.load_model('best_model.h5',compile=False)
    x_pre=preprocess('static\pre.png')
    pred_img = model.predict([x_pre])
    threshold = 0.5
    pred_img_binary = (pred_img > threshold).astype(np.uint8)

        # Save the processed prediction as an image
    output_path = "./static/predicted_change_map.png"
    cv2.imwrite(output_path, pred_img_binary[0, :, :, 0] * 255)
    return render_template('predict.html', result=True, change_map='predicted_change_map.png')
    

        

if __name__=="__main__":
    app.run(port=3000,debug=True)