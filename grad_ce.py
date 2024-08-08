from pprint import pprint
import pandas as pd
from grad import *

# i think it was this dataset: https://www.kaggle.com/datasets/kobymike2020/noisy-xor/data
csv = pd.read_csv("XOR.csv")
train_values = []
eval_values = []
split_pos = 0.8 * len(csv.index)
for pos, i in csv.iterrows():
    r = ((i["X0"], i["X1"]), i["XOR"])
    if pos >= split_pos:
        eval_values.append(r)
    else:
        train_values.append(r)

      
a = Linear(2, 10)
act = Tanh
# b = Linear(10, 1)
b = Linear(10, 2)
# outact = Sigmoid
outact = logSoftmax

optimiser = Adam([a.weights(), b.weights()])
# optimiser = Optimiser([a.weights(), b.weights()], epsilon)

def equation(x):
    x = a.forward(x)
    for i in range(len(x)):
        x[i] = act(x[i])
    x = b.forward(x)
    # x = outact(x[0])
    x = outact(x)
    return x

for i in range(1000):
    sample = random.sample(train_values, 100)
    # sample = train_values
    y_guess = [equation([Value(x) for x in x]) for x, _ in sample]
    # print(y_guess)

    # train
    train_loss = NLLLoss(y_guess, [int(y) for _, y in sample])
    optimiser.step()
    
    # evaluate
    eval_guess = [equation([Value(x) for x in x]) for x, _ in eval_values]
    # eval_loss, _ = BCELoss(y_guess, [Value(y) for _, y in eval_values])
    eval_loss = NLLLoss(y_guess, [int(y) for _, y in eval_values])
    optimiser.zero_gradients()
    
    # train_acc = sum(((guess.val >= 0.5) == (y >= 0.5) for guess, (_, y) in zip(y_guess, sample))) / len(sample)
    train_acc = sum((argmax([g.val for g in guess]) == int(y) for guess, (_, y) in zip(y_guess, sample))) / len(sample)
    # eval_acc = sum(((guess.val >= 0.5) == (y >= 0.5) for guess, (_, y) in zip(eval_guess, eval_values))) / len(eval_values)
    eval_acc = sum((argmax([g.val for g in guess]) == int(y) for guess, (_, y) in zip(eval_guess, eval_values))) / len(eval_values)
    
    # if eval_loss < 0.01:
        # print("breaking early")
        # print(i, train_loss, eval_loss)
        # break
    # if i % 100 == 0:
    print(train_acc, eval_acc, train_loss, eval_loss)


# $ python grad_ce.py
# 0.46 0.47 0.9928205222850904 0.8972358580182509
# 0.49 0.53 1.235896095625925 1.413356864473591
# 0.48 0.485 1.0980825285743514 1.0962073148166545
# 0.52 0.53 1.1050770357816622 1.2999730001629184
# 0.45 0.525 0.8829519303089107 0.8969005002722337
# 0.58 0.53 1.054749078717359 1.3024149044230529
# 0.49 0.665 0.7620819521518232 0.8078918744340268
# 0.69 0.585 0.7224296108718227 0.9659650648390713
# 0.54 0.715 0.6585165861694164 0.8225158872959851
# 0.77 0.535 0.5892155723926833 0.8818574877574272
# 0.47 0.735 0.6068055766520914 0.7503915694735991
# 0.77 0.605 0.5672500236466769 0.8093787735362615
# 0.61 0.715 0.5498992579356462 0.7637978016215443
# 0.69 0.535 0.6716897710304466 0.8091643678863476
# 0.5 0.63 0.7490127059913199 0.9432629027234164
# 0.58 0.53 0.8986862215460079 1.1239314657140356
# 0.44 0.64 0.8951908485416448 0.9218754062690615
# 0.65 0.75 0.9062283416955043 1.2994646108897985
# 0.76 0.715 0.5631611975150665 0.8184388100359944
# 0.78 0.75 0.5483226260994598 0.9438216413884268
# 0.72 0.735 0.5105652574417523 0.8864791707614644
# 0.78 0.76 0.4641697781828602 0.9239064309699669
# 0.77 0.74 0.44172694242958166 0.8617936416536557
# 0.73 0.755 0.47190263649696773 0.8145404361339922
# 0.7 0.7 0.506523617334008 0.8767030170825845
# 0.73 0.74 0.5133699532075261 1.1644464303432644
# 0.71 0.725 0.4589431097285507 0.8058874290812069
# 0.75 0.73 0.47916741307669286 0.9641726627739751
# 0.72 0.715 0.46448528367228803 0.8359761721509676
# 0.76 0.78 0.45652471141970663 1.1461998848945025
# 0.72 0.715 0.41559742093946256 0.9840125327104589
# 0.71 0.745 0.48664280322219483 1.3001410482192162
# 0.77 0.715 0.41290160437534235 1.0868339063020105
# 0.77 0.935 0.3915178590715409 1.1157734515441655
# 0.96 0.835 0.26778693188461505 1.1095108605773845
# 0.86 0.935 0.2827216052021647 1.0783802491987509
# 0.95 0.805 0.2703456017657999 1.0572962722533885
# 0.84 0.885 0.2985197084010588 1.09066864473987
# 0.88 0.755 0.2930625181023677 1.3180067999793326
# 0.81 0.99 0.24346368290336934 1.5857996112479307
# 1.0 0.995 0.1670983239772751 1.2345768543709186
# 0.98 0.995 0.16340276259248568 1.3494756754246218
# 1.0 1.0 0.14854813802637273 1.0820053690926716