import random
from engine import Value

# Neuron is one cell basically
class Neuron:

    # nin is neuron input
    def __init__(self, nin, label=''):
        self.label = label
        # for every neuron input a neuron has, it contains each weight, and one bias for the whole neuron
        # initially weight and bias is random
        self.w = [Value(random.uniform(-1,1), label=f"{label}.w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label=f"{label}.bias")
        self.parameters = self._parameters()


    # this gets run when we call a neuron object
    # n = Neuron(3)
    # calling n() will execute this which basically calculate the output of the nueron
    # we use tanh to activate the function basically squashing it
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act.label = f"{self.label}.sum"
        out = act.tanh()
        out.label = f"{self.label}.out"
        return out

    # in backpropagation we only want to change usually the biases and weights
    # this function returns them
    def _parameters(self):
        return self.w + [self.b]

# Layer is group of neurons
class Layer:

    def __init__(self, nin, nout, label=''):
        # were creating neurons based on the number of output we want
        # and we tell how many inputs does those neurons take
        self.neurons = [Neuron(nin, label=f"{label}.n{i}") for i in range(nout)]
        self.parameters = self._parameters()

    def __call__(self, x):
        # were just calling the _call__ of neurons
        outs = [n(x) for n in self.neurons]
        # since outs is an array
        # if it contains only one output we return the value itself
        # but if it contains many neurons then return the list
        return outs[0] if len(outs) == 1 else outs

    def _parameters(self):
        # get all the weight and biases of each neuron in a layer
        return [p for neuron in self.neurons for p in neuron.parameters]


# MLP or Multi Layer Perceptron
# we can now build on top of the Layer class to actually implement a menaingful neural networks
class MLP:

    # we take number of inputs and nouts array of layers we want
    # let say MLP(3, [4, 4, 1])
    # this is a 3 neuron input layer
    # 4 neuron hidden layer 1
    # 4 neuron hidden layer 2
    # one output neuron
    def __init__(self, nin, nouts):

        # flatten the inputs and output counts
        sz = [nin] + nouts
        # using the above sample
        # sz = [3, 4, 4, 1]
        # we create a layer of 3(i) with output of 4(i+1)
        self.layers = [Layer(sz[i], sz[i+1], label=f"layer{i}") for i in range (len(nouts))]
        # i cached the parameters since were not really cahing the architecture or network atm
        # but maybe when we start doing other shts
        self.parameters = self._parameters()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _parameters(self):
        return [p for layer in self.layers for p in layer.parameters]

    def forward(self, xs):

        ypred = [self(x) for x in xs]

        return ypred

    def loss(self, ys, ypred, type="MSE"):

        match type:
            case "MSE":
                loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
                return loss
            case _:
                raise ValueError(f"{type} loss doesn't exist")
                



    # xs are the inputs
    # ys are the outputs
    # epoch is how many iterations
    # lr is the learning rate, or by how much do we adjust the weights and bias
    def learn(self, xs, ys, epoch, lr):

    

        for k in range(epoch):
            # we feed the x input to the neural net
            ypred = self.forward(xs)
    
            # for now im gonna use mean square error
            # basically we subtract the prediction to the real value
            # to get the error and we square that to eliminate negative values
            # we basically only want to go positive and our goal is to lower it to 0
            loss = self.loss(ys, ypred)
            
            self.zero_grad()
    
            # we back propagate now
            loss.backward()

            # we adjust the weights and biases now
            # going back gradient just points on the direction on where can we increase the value
            # but in this case we want to actually lower it so were going to the oposite of the gradient
            # thats why we multiply the negation of learning rate to it,
            # so basically go to the opposite side of the gradient with lr step
            for p in self.parameters:
                p.data += (-lr) * p.grad
            
            if k % 100 == 0 or k == epoch - 1:
                print(f"epoch {k}: {loss.data}")

        ypred = self.forward(xs)
        loss = self.loss(ys,ypred) # for vis

        return ypred, loss
        


    # zero grad basically just reset every gradient of eveyr nuerons iterations has clean slate
    def zero_grad(self):

        for p in self.parameters:
            p.grad = 0.0


