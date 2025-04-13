import numpy as np
import random
import math

def activation(x): #activation function
    return np.tanh(x)

def act_dev(x): #activation derivative used in backpropagation
    return 1-x**2

#hyperparameters
layers = 5 #number of hidder layers
hidden_size = 22 #size of the first hidden layer
taper = 1 #coefficient that changes the size of subsequent hidden layers
epochs = 1 #number of times the program goes through all iterations
iterations = 500
alpha = 0.00003 #learning rate

#initailize arrays
error_pool = np.empty((epochs))
layer_in = np.zeros((1,3))
layer_out = np.zeros((1,3))
delta_out = np.zeros((1,3))

for j in range(epochs):
    for x in range(1,(layers+1)): #dynamically create layer and delta arrays based on hyperparameters
        layer_name = f'layer_{x}'
        layer_value = np.zeros((1,int(math.ceil(hidden_size*taper**(x-1)))))
        locals()[layer_name] = layer_value
        delta_name = f'delta_{x}'
        delta_value = np.zeros((1,int(math.ceil(hidden_size*taper**(x-1)))))
        locals()[delta_name] = delta_value

        if x == 1: #dynamically assign random connection weights
            weight_in_1 = 2*np.random.random((3,hidden_size))-1
        else:
            weight_name = f'weight_{x-1}_{x}'
            weight_value = 2*np.random.random((int(math.ceil(hidden_size*taper**(x-1))),int(math.ceil(hidden_size*taper**(x)))))-1
            locals()[weight_name] = weight_value
        weight_name = f'weight_{layers}_out'
        weight_value = 2*np.random.random((int(math.ceil(hidden_size*taper**(x-1))),3))-1
        locals()[weight_name] = weight_value

    for y in range(iterations):
        error = 0
        for inc in range(1000):
            rand_in = [random.randrange(1,101),random.randrange(1,101)] #randomly generate base and height
            base = max(rand_in)
            height = min(rand_in)
            shape_select = random.randint(0,2) #randomly select shape type (rectangle, triangle, circle)

            if shape_select == 0: #assign area based on shape type with a small random error
                A = base*height*(1+(2*random.random()-1)/100)
            elif shape_select == 1:
                A = 0.5*base*height*(1+(2*random.random()-1)/100)
            else:
                A = np.pi*base*height*(1+(2*random.random()-1)/100)
            layer_in[0,0],layer_in[0,1],layer_in[0,2] = base,height,A
            layer_in /= 100 #normalize input

            for z in range(1,(layers+1)): #forward propagation
                if z == 1:
                    layer_1 = activation(np.dot(layer_in,weight_in_1))
                else:
                    alayer = f'layer_{z}'
                    xlayer = f'layer_{z-1}'
                    xweight = f'weight_{z-1}_{z}'
                    locals()[alayer] = activation(np.dot(locals()[xlayer],locals()[xweight])) #activation function

            xlayer = f'layer_{layers}'
            xweight = f'weight_{layers}_out'
            layer_out = 1/(1+np.exp(-1*np.dot(locals()[xlayer],locals()[xweight]))) #output layer activation function
            actual = np.zeros((1,3))
            actual[0,shape_select] = 1
            error += np.sum((layer_out-actual)**2) #calculate error
            delta_out = layer_out-actual

            for k in reversed(range(1,(layers+1))): #backpropagation
                if k  ==  layers:
                    xdelta = f'delta_{k}'
                    xweight = f'weight_{k}_out'
                    xlayer = f'layer_{k}'
                    locals()[xdelta] = np.dot(delta_out,locals()[xweight].T)*act_dev(locals()[xlayer])
                else:
                    xdelta = f'delta_{k}'
                    ydelta = f'delta_{k+1}'
                    xweight = f'weight_{k}_{k+1}'
                    xlayer = f'layer_{k}'
                    locals()[xdelta] = np.dot(locals()[ydelta],locals()[xweight].T)*act_dev(locals()[xlayer])

            for t in range(layers+1): #update weights
                if t  ==  0:
                    xweight = f'weight_in_1'
                    xlayer = f'layer_in'
                    xdelta = f'delta_1'
                    locals()[xweight] -= alpha*locals()[xlayer].T.dot(locals()[xdelta])
                elif t  ==  layers:
                    xweight = f'weight_{t}_out'
                    xlayer = f'layer_{t}'
                    xdelta = f'delta_out'
                    locals()[xweight] -= alpha*locals()[xlayer].T.dot(locals()[xdelta])
                else:
                    xweight = f'weight_{t}_{t+1}'
                    xlayer = f'layer_{t}'
                    xdelta = f'delta_{t+1}'
                    locals()[xweight] -= alpha*locals()[xlayer].T.dot(locals()[xdelta])

        print(f'trial: {y}, error: {error}')

n_test = 1000
correct = 0

for it in range(n_test): #test the network
    rand_in = [random.randrange(1,101),random.randrange(1,101)]
    base = max(rand_in)
    height = min(rand_in)
    shape_select = random.randint(0,2) #select shape

    if shape_select == 0: #assign area
        A = base*height#*(1+(2*random.random()-1)/100)
    elif shape_select == 1:
        A = 0.5*base*height#*(1+(2*random.random()-1)/100)
    else:
        A = np.pi*base*height#*(1+(2*random.random()-1)/100)
    layer_in[0,0],layer_in[0,1],layer_in[0,2] = base,height,A #input values
    layer_in /= 100 #normalize

    for z in range(1,(layers+1)): #forward propagation
        if z == 1:
            layer_1 = activation(np.dot(layer_in,weight_in_1))
        else:
            alayer = f'layer_{z}'
            xlayer = f'layer_{z-1}'
            xweight = f'weight_{z-1}_{z}'
            locals()[alayer] = activation(np.dot(locals()[xlayer],locals()[xweight])) #activation function

    xlayer = f'layer_{layers}'
    xweight = f'weight_{layers}_out'
    layer_out = 1/(1+np.exp(-1*np.dot(locals()[xlayer],locals()[xweight]))) #output layer activation function
    actual = np.zeros((1,3))
    actual[0,shape_select] = 1 #output

    if np.argmax(layer_out) == np.argmax(actual): #check if output is correct
        correct += 1

print(f'Correct classifications: {round(correct*100/n_test,2)}%')

print('Program Complete')
