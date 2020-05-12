from imageai.Prediction import ImagePrediction
import os

# This is the directory that we're working in, which the test images are in too
execution_path=os.getcwd()

# Instantiate the image prediction
prediction = ImagePrediction()
# Set the model that we want to use (copied from the documentation and amended to the model we want to use
prediction.setModelTypeAsSqueezeNet()
# Import the model from the ImageAI GitHub repo and into the project directory and set the model path
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
# Load the model
prediction.loadModel()

# Now we run our predictions - result_count is the number of predictions we want the model to give us
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
