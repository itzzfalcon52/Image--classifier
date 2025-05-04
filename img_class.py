import torch
import os
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


if torch.cuda.is_available():
    device = torch.device("cuda") #for windows
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps") #for mac
else:
    device = torch.device("cpu")

#1.Data Transforms
transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          (0.5,0.5,0.5),
          (0.5,0.5,0.5)
      )

])
#2.Loaders
root = os.path.join(os.getcwd(), 'data')
train_set=torchvision.datasets.CIFAR10(root=root,train=True,download=True,transform=transform)
test_set=torchvision.datasets.CIFAR10(root=root,train=False,download=True,transform=transform)

train_loader=torch.utils.data.DataLoader(train_set,batch_size=10,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=10,shuffle=False)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#3.visualisation of datasets
fig, axes=plt.subplots(1,10,figsize=(12,3))
for i in range(10):
    image=train_loader.dataset[i][0].permute(1,2,0)
    denormalized_image=image/2+0.5
    axes[i].imshow( denormalized_image)
    axes[i].set_title(classes[train_loader.dataset[i][1]])
    axes[i].axis('off')


#4.CNN model
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3)
        self.conv2=nn.Conv2d(64,128,3)
        self.pool=nn.MaxPool2d(2,stride=2)

        self.fc1=nn.Linear(128*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)


    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.log_softmax(self.fc3(x),dim=1)
        return x
net=ConvNeuralNet()
net.to(device)
MODEL_PATH = "cifar_cnn.pth"
#5.load the saved module or train a new one
if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded pretrained model from {MODEL_PATH}, skipping training.")
else:
#6.training module
  loss_function=nn.NLLLoss()
  optimizer=optim.Adam(net.parameters(),lr=0.001)

  epochs=10
  for epoch in range(epochs):

     running_loss=0.0
     for i,data in enumerate(train_loader):
        inputs,labels=data[0].to(device),data[1].to(device)

        optimizer.zero_grad()
        outputs=net(inputs)
        loss=loss_function(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
        if i % 2000 == 1999:
            running_loss=0.0
  torch.save(net.state_dict(), MODEL_PATH)
  print(f"Model trained and saved to {MODEL_PATH}")

#7.Confusion Matrix 
net.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute and display confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()



#8.Final predicted image visualisation grpahs
def view_classification(image,probabilites):

    probabilites=probabilites.data.numpy().squeeze()

    fig, (ax1,ax2)=plt.subplots(figsize=(6,9),ncols=2)
    image=image.permute(1,2,0)
    denormalized_image=image/2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10),probabilites)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title("Class Probability")
    ax2.set_xlim(0,1.1)
    plt.tight_layout()

images,_=next(iter(test_loader))
net.eval()
for i in range(10):  
    image = images[i]
    batched_image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        log_probabilities = net(batched_image)
    
    probabilities = torch.exp(log_probabilities).squeeze().cpu()
    
    view_classification(image, probabilities)

plt.show()




    



        
