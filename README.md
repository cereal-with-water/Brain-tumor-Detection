# Brain Tumor Detector
#### For this project, we we will create a service as a web app using Streamlit.<br />
#### Image classification is a computerized visual recognition model algorithm that can assign images to certain predefined classes based on the contents of an image file.<br />
#### This will be used to allow the program we will be servicing to perform classification and sorting operations on image data.<br />
#### The classifier we will use here will utilize a pre-trained model using Google Teachalbe Machine and will be written in Python.<br />https://teachablemachine.withgoogle.com/train


## Requirements/Environments
- Python 3 is required

### Library requirements
- Tensorflow
- Keras
- Random
- Time
- Plot
- streamlit

## Creating an Image Classifier Model
~~We are allowed to use this tool right? still need to confirm thorugh him~~<br /><br /><br />
1. Google Teachable Machine allows you to create projects with images as well as audio or poses In this case, we'll select "Images Project" as our image classification project.<br />https://teachablemachine.withgoogle.com/train<br /><br />

2. You need a dataset for training. Download the dataset and unzip the Brain Tumor Detection dataset from Kaggle.<br />https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection<br /><br />

3. On the Image Classification page, create a class to label brains with and without tumors, upload images to that class, and let it train. <br /><br />
   <img src = "https://github.com/NoMoreError/5820-Final-Project/assets/113921954/5cd96cf1-58b3-491a-90f4-52f3f4bfa2e8" width = "600" height = "600"> <img src = "https://github.com/NoMoreError/5820-Final-Project/assets/113921954/9e9bfc77-e988-432e-b73b-7c7e8e9b6c34" width = "600" height = "600">
   <img src = "https://github.com/NoMoreError/5820-Final-Project/assets/113921954/9e9bfc77-e988-432e-b73b-7c7e8e9b6c34" width = "600" height = "600">
   <img src = "https://github.com/NoMoreError/5820-Final-Project/assets/113921954/9bde33e5-d88f-4055-8854-0fa4a09c43b7" width = "600" height = "600">




5. Once it's done, download the trained file (a weighted file with the .h5 extension) and use it to provide a service.<br />
Follow the video link below and try it out.<br />https://www.youtube.com/watch?v=bDsR1K53Ew0<br /><br />


6. If you look in "Export", you'll see the example source as follow.<br />
We're going to build on this and make stramlit process the model and imag parts of the source below seperately.**<br /><br />


`model = tensorflow.keras.models.load_model('keras_model.h5')`<br />
`image = image.open('test_photo.jpg`)


## Training the model

## Results

## Final Notes

## Research References
All reference links used for the project:<br />
https://www.nature.com/articles/s41598-024-52823-9<br />
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9854739/
