import pandas as pd
import numpy as np
import pickle as pkl

train_data = pd.read_csv('D:\\UniversityOfWaterloo\\Courses\\657_ToolsOfIntelligentSystemDesign\\Assignment1\\train_data_orig.csv').values
train_lables = pd.read_csv('D:\\UniversityOfWaterloo\\Courses\\657_ToolsOfIntelligentSystemDesign\\Assignment1\\train_label_orig.csv').values

print(len(train_data))
split_val = round(0.8*len(train_data))
indices = np.random.permutation(train_data.shape[0])

training_idx, test_idx = indices[:split_val], indices[split_val:]
training_data, testing_data = train_data[training_idx,:], train_data[test_idx,:]
training_label_data, testing_label_data = train_lables[training_idx,:], train_lables[test_idx,:]

print(train_data.shape)
print(training_data.shape)
print(testing_data.shape)

class SimpleMLP():
    output_layer1 = 0
    output_layer2 = 0
    
    def __init__(self, no_hidden_layers, hidden_layer, input_data, output_data):
        
        self.input_layer = input_data.shape[1]
        self.no_hidden_layers = no_hidden_layers #1
        self.hidden_layer = hidden_layer #770
        self.output_layer = 4
        
        self.input_data = input_data
        self.output_data = output_data
        
        #Initialize weights for layer 1
        self.weights_layer1 = np.random.randn(self.input_layer, self.hidden_layer)
        #bias for each neuron for layer 1
        self.bias_layer1 = np.zeros(hidden_layer)
        
        #Initialize weights for output layer 1
        self.weights_layer2 = np.random.randn(self.hidden_layer, self.output_layer)
        #bias for each neuron for layer 2
        self.bias_layer2 = np.zeros(output_data.shape[1]) 
    
# create the instance of the NN model
mlpModel = SimpleMLP(1, 770, training_data, training_label_data) 
mlpModelTest = SimpleMLP(1, 770, testing_data, testing_label_data)

def sigmoid(x):
    output = 1 / (1 + np.exp(-x) )
    return output

def sigmoid_derivative(x):
    output = (x*(1.0-x))
    return output

def ReLU(x):
    return max(0.0, x)

def soft_max(x):
    exponent = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponent/np.sum(exponent, axis=1, keepdims=True)

def soft_max_two(x):
    exponent = np.exp(x)
    return exponent/np.sum(exponent, axis=1, keepdims=True)

def error_loss(predicted, actual):
    no_samples = actual.shape[0]
    res = predicted - actual
    return(res/no_samples)

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def forward_pass(mlp_model, batch_no, batch_size):
    
    #input Layer 1
    mlp_model.output_layer1 = np.dot(mlp_model.input_data[batch_no*batch_size: (batch_no+1)*batch_size], mlp_model.weights_layer1) + mlp_model.bias_layer1
    mlp_model.output_layer1 = sigmoid(mlp_model.output_layer1)
    
    #ouput layer 2
    mlp_model.output_layer2 = np.dot(mlp_model.output_layer1, mlp_model.weights_layer2) + mlp_model.bias_layer2
    mlp_model.output_layer2 = soft_max_two(mlp_model.output_layer2)
    
def backward_pass(mlp_model, batch_no, batch_size):
    learning_rate = 0.025
    
    error_layer2 = error_loss(mlp_model.output_layer2, mlp_model.output_data[batch_no*batch_size: (batch_no+1)*batch_size])
    output2_delta = np.dot(error_layer2, mlp_model.weights_layer2.T)
    error_layer1 = output2_delta * sigmoid_derivative(mlp_model.output_layer1)
    
    mlp_model.weights_layer2 = mlp_model.weights_layer2 - learning_rate * np.dot(mlp_model.output_layer1.T, error_layer2)
    mlp_model.bias_layer2 = mlp_model.bias_layer2 - learning_rate * np.sum(error_layer2, axis=0, keepdims=True)
    mlp_model.weights_layer1 = mlp_model.weights_layer1 - learning_rate * np.dot(mlp_model.input_data[batch_no*batch_size: (batch_no+1)*batch_size].T, error_layer1)
    mlp_model.bias_layer1 = mlp_model.bias_layer1 - learning_rate * np.sum(error_layer1, axis=0)
    
def check_accuracy(mlp_model, no_epoch):
    model_accuracy = 0
    accuracy_percent = 0
    mlpModelTest.weights_layer1 = mlp_model.weights_layer1
    mlpModelTest.weights_layer2 = mlp_model.weights_layer2
    forward_pass(mlpModelTest, 0, mlpModelTest.output_data.shape[0])
    for a,b in zip(mlpModelTest.output_layer2, mlpModelTest.output_data):
        if a.argmax() == b.argmax():
            model_accuracy = model_accuracy +1
    accuracy_percent = (model_accuracy/mlpModelTest.output_data.shape[0])*100
    print("Training accuracy at epoch %d is %f :",no_epoch,accuracy_percent)
    return(accuracy_percent)

def train_nn(mlp_model):
    no_epochs =20
    batch_size = 100
    total_batches = (mlp_model.input_data.shape[0] - (mlp_model.input_data.shape[0] % batch_size))/batch_size
    for i in range(0, no_epochs):
        for j in range(0, int(total_batches)):
            forward_pass(mlp_model, j, batch_size)
            backward_pass(mlp_model, j, batch_size)
        check_accuracy(mlp_model, i)
    with open('weightsOneV1.pkl','wb') as f:
        pkl.dump(mlpModel.weights_layer1, f)
    with open('weightsTwoV2.pkl','wb') as f2:
        pkl.dump(mlpModel.weights_layer2, f2)
    with open('biasOne.pkl','wb') as f3:
        pkl.dump(mlpModel.bias_layer1, f3)
    with open('biasTwo.pkl','wb') as f4:
        pkl.dump(mlpModel.bias_layer2, f4)

train_nn(mlpModel)

            
            
            
            
            
            
            
            
            
        
    
        