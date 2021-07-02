# -Transfer-learning-

### Freeze network parameters, i.e. not updated during training

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.fc1 = nn.Squential(
                                 nn.Linear(),
                                 nn.Linear(),
                                 ReLU(inplace=True),
                                )
        self.classifier = nn.Linear()
```
**Set the layer requires_grad attribute to False to freeze the layer parameters**
Example:
```python
for param in layer.parameters():
	param.requires_gard = False

# if we will to freeze the parameters of self.fc1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.fc1 = nn.Squential(
                                 nn.Linear(),
                                 nn.Linear(),
                                 ReLU(inplace=True),
                                )
        for param in self.fc1:
        	param.requires_gard = False
        self.classifier = nn.Linear()
```
**We also need to tell the optimizer which needs to be updated and which does not. This step is very important**

```python
optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
```
**We can also freeze most of the layers**
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.fc1 = nn.Squential(
                                 nn.Linear(),
                                 nn.Linear(),
                                 ReLU(inplace=True),
                                )
        for param in self.parameters():
            param.requires_grad = False
        #这样for循环之前的参数都被冻结，其后的正常更新。
        self.classifier = nn.Linear()
```
