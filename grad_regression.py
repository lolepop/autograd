from pprint import pprint
from grad import *

# train_values = [(20, 176), (-8, -20), (-100, -664)]
train_values = [(x, 12*x**2 - 3*x + 8) for x in [-5,-4,-3,-2,-1,0,1,2,3,4,5]]

a = Value(0)
b = Value(0)
c = Value(0)

optimiser = Adam([a, b, c])

def equation(x):
    # return m.mul(x).add(c)
    return a*x**2 + b*x + c

for i in range(1000):
    sample = random.sample(train_values, 4)
    # sample = train_values
    y = [equation(Value(x)) for x, _ in sample]

    loss = MSELoss(y, [Value(y) for _, y in sample])
    # for yy, d in zip(y, dloss):
        # yy.backward(d)
    optimiser.step()
    
    # doing it like this is stupid but whatever
    if loss < 20:
        print("breaking early")
        print(i, loss)
        break
    # if i % 100 == 0:
    print(loss)

pprint(vars(a))
pprint(vars(b))
pprint(vars(c))

# ok i think we get the picture
# $ python grad_regression.py
# 35006.5
# 7393.660046998767
# 6924.792757046471
# 281.57926629179593
# 86.82264973971141
# 3259.4775371587316
# 671.0081468390507
# 281.33285329958625
# 220.0506431617995
# 29.78992133208447
# 3320.7782569120573
# 71.72744492654304
# 6374.827154119472
# 567.876603953305
# 435.81434252835754
# 1771.0787971814023
# 61.489011346289786
# 2000.007711753256
# 68.81043143098829
# breaking early
# 19 9.586937751423868
# {'gradient': 0, 'parents': [], 'val': 13.551039654418457}
# {'gradient': 0, 'parents': [], 'val': -3.784826766700124}
# {'gradient': 0, 'parents': [], 'val': 12.86250681111875}