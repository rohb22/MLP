import math
# Data strucuture to store scalar values, that support diff operations and backpropagation

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None # function to backpropagate
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # this would be the text presentation of the value class
    def __repr__(self): 
        return f"Value(data={self.data})"

    # this would be the function to run when u add a Value object
    def __add__(self, other):
        # if value object is added with an integer or other data type we wrap it to Value class first
        # this would work for cases like ValueObj + 3
        other = other if isinstance(other, Value) else Value(other)

        # the result of the addition would be a Value object also
        out = Value(self.data + other.data, (self, other), '+')

        # this computes how the gradient of out is propagated to self and other
        def _backward():
            # we multiply by 1 since the derivative of a + b respect to a is just 1
            # we multiply by out.grad because of chain rule
            # we add the gradient because it needs to accumulate in cases where the same Value object is used
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # in cases where Value + otherd data type the __add__ would work given that python would express the Value object first
    # but when the case become other data type, + Value, the other data type __add__ would be called
    # so here we use __radd__ as a fallback so python knows what to do
    # basically were saying that if the operation failed, our Value class knows how to handle it
    # which is basically just switching the values position in the equeation
    def __radd__(self, other):
        return self + other
        


    # same case as addition, just for multiplication the only thing would change is how we take the gradient
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():

            # if we have a * b and we need to get the derivative respect to a it becomes 1*b or just b
            # so the gradient is the the other value
            # again multiply by out grad because of chain rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # for subtraction we implemented it via negating the other operand
    # for simplicity sake since were gonna be implementing negation in the first place
    def __sub__(self, other):
        return self + (-other)

    # this would be called in cases like 2 - Value 
    # order matters in subtraction
    def __rsub__(self, other):
        return other + (-self)

    # we simplified things that most of the operations could be defined by the addition and multiplication
    # so we dont need to implement extra stuff
    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)) # we only support where the exponent is an integer or float

        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():

            # to get the derivative of self**other using power rule
            # we get other * self**other-1
            # and to get the gradient we multiply again by the out.grad
            # x**3 = 3 * x ** 2
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        # let say 3/2 its the same as 3 * 1/2
        # this simplify the definition
        return self * other ** - 1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # to get the gradient, the derivative of e^x is just e^x
            # since out is already e^x we just multiply by the out.grad cause chain rule
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    # this is the activation function were gonna use, for now atleast
    def tanh(self):
        n = self.data
        # the definition of tanh is  (e^n - e^-n)  / (e^n + e^-n)
        # if we multiply that by e^n / e^n
        # the simplification of tanh is
        # (e^2n - 1) / (e^2n + 1)
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # using identities for sinh(n)/cosh(n)
            # we can come up with the derivative of tanh
            # which is 1 - tanh^2(n)
            # since t here is already the tanh we just square it
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    # i gotta learn more about this one
    # but basically were creating a graph (Directed Acyclic Grapg DAG) in such a way that
    # when we backpropagate, every node receives all the gradient contributions 
    # from nodes after it before we propagate its own gradient to its children.  
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        # our base case the output has a gradient of 1
        # The output  has a gradient of 1 with respect to itself.
        self.grad = 1
        build_topo(self)

        # we backpropagate startign from the last node
        for node in reversed(topo):
            node._backward()