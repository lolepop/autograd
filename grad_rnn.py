from pprint import pprint
import pandas as pd
from grad import *
        
train_values = ["testthingy"]
token_list = ["\n"] + [chr(i) for i in range(32, 127)]
token_mapping = { v: i for i, v in enumerate(token_list) }

a = RNNLayer(len(token_list), len(token_list), 20)

optimiser = Adam([a.weights()])

def tokenise(s):
    return [token_mapping[c] for c in s]

# convert onehot encoded
def encode(tokenised):
    return [[Value(int(i == token)) for i in range(len(token_list))] for token in tokenised]

def equation(x, i=1):
    x = a.forward(x, i)
    return x

tokenised_train_values = [tokenise(v) for v in train_values]
encoded_train_values = [encode(v) for v in tokenised_train_values]
for i in range(5):
    # sample = random.sample(train_values, 100)
    # sample_index = random.randrange(len(train_values))
    sample_index = 0
    testsample = encoded_train_values[sample_index]
    y_guess = equation(testsample)
    
    # train
    train_loss = NLLLoss([y_guess[:-1]], [tokenised_train_values[sample_index][1:]], multi=True)
    optimiser.step()
    
    print(train_loss)
    

s = ""
inp = "t"
for i in equation(encode(tokenise(inp)), 20):
    newtoken = [v.val for v in i]
    s += token_list[argmax(newtoken)]
print(inp + s)

# it does in fact work but just running this alone took around 30s
# i guess this is far as i get so i wont bother cleaning up the rest of the code
# $ python grad_rnn.py
# 5.943790499435092
# 3.8798026913376256
# 2.452456681163106
# 1.4012119955988023
# 0.8083029989175551
# thingyngyngyngyjhingy