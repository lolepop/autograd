import random
import math

def clamp(a, low, high):
    if a > high:
        return high
    elif a < low:
        return low
    return a

# coerce into value type for raw nums
def auto_convert(o):
    return o if isinstance(o, Value) else Value(o)

def argmax(arr):
    mp = (0, arr[0])
    for i, v in enumerate(arr):
        if v > mp[1]:
            mp = (i, v)
    return mp[0]

def flatten_weights(weights):
    def inner(param):
        if isinstance(param, list):
            for p in param:
                yield from inner(p)
        else:
            yield param
    return inner(weights)

class Value():
    def __init__(self, val, parents=[]):
        self.val = val
        self.parents = parents
        self.gradient = 0
    
    def trunc(self):
        return Value(self.val)
        
    def __add__(self, b):
        return self.add(b)
    
    def __sub__(self, b):
        return self.sub(b)
    
    def __truediv__(self, b):
        return self.div(b)
    
    def __mul__(self, b):
        return self.mul(b)
        
    def __pow__(self, b):
        return self.pow(b)
        
    def __neg__(self):
        return self.neg()
    
    def add(self, b):
        return Add(self, auto_convert(b))
    
    def sub(self, b):
        return Sub(self, auto_convert(b))
        
    def mul(self, b):
        return Mul(self, auto_convert(b))
    
    def div(self, b):
        return Div(self, auto_convert(b))
        
    def pow(self, b):
        return Pow(self, auto_convert(b))
        
    def neg(self):
        return Neg(self)
    
    def backward(self, acc):
        acc = clamp(acc, -1, 1)
        self.gradient += acc
        
    def apply_gradient(self, epsilon):
        self.val -= self.gradient * epsilon
        self.zero_gradients()
        
    def zero_gradients(self):
        self.gradient = 0
    
    def __repr__(self):
        return str(self.val)
    
    def __str__(self):
        return str(self.val)

class Add(Value):
    def __init__(self, a, b):
        rawVal = a.val + b.val
        super().__init__(rawVal, (a, b))
    
    def backward(self, acc):
        da, db = acc, acc
        a, b = self.parents
        a.backward(da)
        b.backward(db)

class Sub(Value):
    def __init__(self, a, b):
        rawVal = a.val - b.val
        super().__init__(rawVal, (a, b))
    
    def backward(self, acc):
        da, db = acc, -acc
        a, b = self.parents
        a.backward(da)
        b.backward(db)

class Mul(Value):
    def __init__(self, a, b):
        rawVal = a.val * b.val
        super().__init__(rawVal, (a, b))
    
    def backward(self, acc):
        a, b = self.parents
        da, db = acc * b.val, acc * a.val
        a.backward(da)
        b.backward(db)

class Div(Value):
    def __init__(self, a, b):
        rawVal = a.val / b.val
        super().__init__(rawVal, (a, b))
    
    def backward(self, acc):
        a, b = self.parents
        da, db = acc / b.val, acc * -a.val / (b.val ** 2)
        a.backward(da)
        b.backward(db)

class Neg(Value):
    def __init__(self, a):
        rawVal = -a.val
        super().__init__(rawVal, (a,))
    
    def backward(self, acc):
        self.parents[0].backward(-acc)

class Pow(Value):
    def __init__(self, a, b):
        rawVal = a.val ** b.val
        super().__init__(rawVal, (a, b))
    
    def backward(self, acc):
        a, b = self.parents
        da = acc * b.val * a.val**(b.val-1)
        a.backward(da)
        if a.val >= 0: # uh how do i deal with negative exponents
            db = acc * a.val**b.val * math.log(a.val + 1e-12)
            b.backward(db)

class Log(Value):
    def __init__(self, a):
        rawVal = math.log(a.val)
        super().__init__(rawVal, (a,))
        
    def backward(self, acc):
        da = acc / self.parents[0].val
        self.parents[0].backward(da)

class Sigmoid(Value):
    def __init__(self, a):
        rawVal = 1 / (1 + math.exp(-a.val))
        super().__init__(rawVal, (a,))
        
    def backward(self, acc):
        da = acc * self.val * (1 - self.val)
        self.parents[0].backward(da)

class Tanh(Value):
    def __init__(self, a):
        rawVal = math.tanh(a.val)
        super().__init__(rawVal, (a,))
        
    def backward(self, acc):
        da = acc * (1 - self.val ** 2)
        self.parents[0].backward(da)

def logSoftmax(vals):
    e = Value(math.e)
    exps = [Log(e ** v) for v in vals]
    sum = Value(0)
    for v in vals:
        sum += e ** v
    sum = Log(sum)
    return [v - sum for v in vals]

def MSELoss(guess, truth):
    # guess = [o.trunc() for o in guess]
    sum = None
    for g, t in zip(guess, truth):
        a = (g - t) ** 2
        sum = (sum + a) if sum else a
    mse = sum / Value(len(guess))
    mse.backward(1)
    # return mse.val, [o.gradient for o in guess]
    return mse.val
    
def BCELoss(guess, truth):
    # guess = [o.trunc() for o in guess]
    sum = Value(0)
    for g, t in zip(guess, truth):
        a = t * Log(g) + (Value(1) - t) * Log(Value(1) - g)
        sum += a
    loss = -sum / len(guess)
    loss.backward(1)
    # return loss.val, [o.gradient for o in guess]
    return loss.val

# guess: list of lists containing value per class [ [class 1 param 1, class 1 param 2, ...], [class 2 param 1, class 2 param 2, ...] ]
# truth: class index (not a mask)
def NLLLoss(guess, truth, multi=False):
    # guess = [[b.trunc() for b in a] for a in guess]
    sum = Value(0)
    for g, t in zip(guess, truth):
        if multi:
            for g, t in zip(g, t):
                prob_correct = g[t]
                sum += prob_correct    
        else:
            prob_correct = g[t]
            sum += prob_correct
    loss = -sum / len(guess)
    loss.backward(1)
    return loss.val

class Linear():
    def __init__(self, input_size, output_size, use_bias=True, init_weight_range=(-1, 1), init_bias_range=(-1, 1)):
        self.use_bias = use_bias
        self.input_size = input_size
        self.output_size = output_size
        self.layer = []
        self.bias = [Value(random.uniform(*init_bias_range)) for i in range(output_size)] if use_bias else []
        for i in range(output_size):
            self.layer.append([])
            for v in range(input_size):
                self.layer[i].append(Value(random.uniform(*init_weight_range)))
    
    def forward(self, x):
        output_matrix = []
        # transposed matrix multiplication
        for i in range(self.output_size):
            output = None
            layer = self.layer[i]
            for v in range(self.input_size):
                out = layer[v] * x[v]
                output = output + out if output else out
            if self.use_bias:
                output = output + self.bias[i]
            output_matrix.append(output)
        return output_matrix

    def apply_gradient(self, epsilon):
        for i in range(self.output_size):
            self.bias[i].apply_gradient(epsilon)
            a = self.layer[i]
            for v in range(self.input_size):
                a[v].apply_gradient(epsilon)
      
    def weights(self):
        return [self.layer, self.bias]

# this works but is so slow i wont bother working on this
class RNNLayer():
    # input and output sizes should be the same for text-text generation
    def __init__(self, input_class_size, output_class_size, hidden_size):
        # TODO: change random init parameters
        self.input_class_size = input_class_size
        self.output_class_size = output_class_size
        self.hidden_size = hidden_size
        # linear part of s
        self.u = Linear(input_class_size, hidden_size, use_bias=False)
        self.w = [Value(random.uniform(-1, 1)) for i in range(hidden_size)]
        self.s_activation = Tanh
        self.v = Linear(hidden_size, output_class_size, use_bias=False)
        self.o_activation = logSoftmax
    
    # x: array of one hot representations. one hot representation is constructed inside this function: [[one hot encoded token 1], ...]
    def forward(self, x, advance_tokens=1):
        total_steps = len(x)
        
        s_prev = [Value(0) for i in range(self.hidden_size)]
        # o: prediction of second token to token at total_steps + 1
        o = []
        for step in range(total_steps + advance_tokens - 1):
            # argmax([i.val for i in o[-1]])
            if step >= total_steps:
                # what is the proper way of doing this
                last_best_token = argmax([i.val for i in o[-1]])
                x_step = [Value(int(i == last_best_token)) for i in range(self.input_class_size)]
            else:
                x_step = x[step]
            # x_onehot = [Value(int(i == x_step)) for i in range(self.input_class_size)]
            x_onehot = x_step
            # Ux + Ws-1
            uw_out = self.u.forward(x_onehot)
            for i in range(self.hidden_size):
                uw_out[i] += self.s_activation(self.w[i] * s_prev[i])
            s_prev = uw_out
            o.append(self.o_activation(self.v.forward(s_prev)))
        return o
            
    def weights(self):
        return [self.u.weights(), self.w, self.v.weights()]

class Optimiser():
    # weights contain every layer weight flattened
    def __init__(self, weights, epsilon=1):
        self.weights = list(flatten_weights(weights))
        self.epsilon = epsilon
        self.ignore_epsilon = False
    
    def zero_gradients(self):
        for w in self.weights:
            w.zero_gradients()
    
    def step(self):
        e = 1 if self.ignore_epsilon else self.epsilon
        for w in self.weights:
            w.apply_gradient(e)

class Adam(Optimiser):
    def __init__(self, weights, lr=0.001, b1=0.9, b2=0.999):
        super().__init__(weights)
        self.ignore_epsilon = True
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.momentum = [0 for i in range(len(self.weights))]
        self.velocity = [0 for i in range(len(self.weights))]
        self.epsilon = 1e-8
        self.timestep = 1
    
    def step(self):
        for i in range(len(self.weights)):
            w = self.weights[i]
            m = self.momentum[i] = self.b1 * self.momentum[i] + (1 - self.b1) * w.gradient
            v = self.velocity[i] = self.b2 * self.velocity[i] + (1 - self.b2) * (w.gradient ** 2)
            mhat = m / (1 - self.b1 ** self.timestep)
            vhat = v / (1 - self.b2 ** self.timestep)
            w.gradient -= self.lr * mhat / (math.sqrt(vhat) + self.epsilon)
        self.timestep += 1
        super().step()

