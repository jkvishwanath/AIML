write a program for Training a Convolutional Neural Network (CNN) to classify whether an image contains a specific person involves using a supervised learning approach
------------------------------------------------------------------------------------------------
dataset/
  person_x/      # contains images of Person X
  not_person_x/  # contains images of others
Images are labeled automatically based on folder names.
======================================================================================

Install python 
download vs code editor
set python path variable. 

TensorFlow has specific compatibility:

TensorFlow Version	Compatible Python Versions
TF 2.15 (latest stable)	Python 3.8 – 3.11
TF 2.13 and earlier	Python 3.7 – 3.10
TF 1.x	Often Python 3.5 – 3.7
No TF support	Python 3.12 (as of May 2025) ❌

python.exe -m pip install --upgrade pip

pip install streamlit tensorflow pillow
pip install scipy 
---------------------------------------------------

In VS Code terminal or any Python IDE:

python train_model.py

After training, you'll get:


person_x_classifier.h5  ✅ (your saved model)

============================================================


How to Run the App

In your terminal or command prompt, run:

==>run this after model - person_x_classifier.h5 got generated . after running train_model.py ran. 
Ensure your model file person_x_classifier.h5 is in the same folder as the Streamlit app script.

streamlit run app.py

This will open the app in your browser where you can upload images and see predictions.

