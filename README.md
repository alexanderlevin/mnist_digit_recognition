# MNIST digit recognition demo

Little webapp for demoing digit recognition using the convolutional neural network.
Originally based on a Tensorflow tutorial, but has since been ported to Keras.

## Set up your environment
1. Create and activate a virtualenv
2. Install the requirements (e.g. `pip install -r requirements.txt`)
within the virtualenv


## Train the model
Run the training batch via, for example,  
```
python -m batch.train_model --num-epochs 5
```

## Start up a local dev server for the webapp
```
flask --app webapp/main run
``` 

This webapp vendors in the Signature Pad javascript library (https://github.com/szimek/signature_pad).
Signature Pad is released under the MIT License.
