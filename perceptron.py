import random
import yaml

# This function loads the inputs from the path-given file
def load_txt_inputs(path):
    with open(path,"r") as file:
        inputs = yaml.safe_load(file)
    return inputs

# This function determines the weight and bias values between the range of 0 to 1 randomly
def weights_bias_determiner(size_of_inputs):
    weights = [ ]
    for i in range(int(size_of_inputs)):
        weights.append(random.uniform(0, 1))
    bias = random.uniform(0, 1)
    return bias, weights

# This is the Rectified Linear Unit (ReLu) activation function
def ReLu(x):
    if x > 0:
        x = x
    else:
        x = 0
    return x 

# Forward propagation through a single perceptron
def perceptron_f_prop(inputs, weights, bias):
    sum = 0
    for m in range(len(weights)):
        sum += inputs[m] * weights[m]
    z = sum + bias
    output = ReLu(z)
    return output

# This function calculates the Mean Squared Error (MSE)
def error_calculator(perdicted, actual):
    MSE = (perdicted - actual)**2
    return MSE

# This function calculates the derivate MSE respect to the predicted output
def error_derivative(predicted, actual):
    mse_d = 2 * (predicted - actual)
    return mse_d

# This functions provides to update the weight values by using gradient descent
def weights_updater(inputs, weights, lr, error_partial_d):
    updated_weights = [ ]
    for u in range(len(inputs)):
       updated_weights.append(weights[u] -(lr * error_partial_d * inputs[u])) 
    return updated_weights

# Main Function to connect all other functions to calculate the output
def main():
    initial_set = [float(x) for x in load_txt_inputs("user_inputs.txt").split(" ")]   # Load the inputs from the file of "user_inputs.txt" which includes the user inputs.
    bias, weights = weights_bias_determiner(len(initial_set))   # Initialize the weights and bias values
    initial_weights = weights   
    size_of_inputs = len(initial_set)   # Get the size of inputs
    lr = 0.01   # Set the Learning Rate
    target_output = initial_set[-1] # Set the target output
    epochs = 4  # Set the epoch value to find a more consistent output
    inputs = initial_set[:size_of_inputs]   # Extract the input values
    
    for e in range(epochs):
        output = perceptron_f_prop(inputs, weights, bias)
        error = error_derivative(output, target_output) # Calculate the derivative of the error with respect to the predicted output
        weights = weights_updater(inputs, weights, lr, error)   # Update weights by using gradient descent
    # Save the updated weights to the file of "updated_weights.txt"
    with open("updated_weights.txt", "w") as f:
        f.write(str(weights))   
    print(f"Initial Weights:\n{initial_weights}\n--------------------\nActual Value: {target_output}\nPredicted Value: {output}\nMean Squared Error: {error_calculator(output, target_output)}\n-------------------- \nUpdated Weights: \n{weights} \nUpdated weights have been saved to updated_weights.txt.") 

# Entry point
if __name__=="__main__":
    main()