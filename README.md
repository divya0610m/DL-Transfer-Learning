# EXPERIMENT 4 :  Developing a Neural Network Classification Model using Transfer Learning
## NAME : DIVYA LAKSHMI M
## REGISTRATION NUMBER : 212224040082


## AIM : 
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

<img width="407" height="109" alt="image" src="https://github.com/user-attachments/assets/007e567b-5ada-41b9-8caf-2170188befff" />


## Neural Network Model
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/06757850-b127-4ffc-899c-119e75e7d581" />

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name: DIVYA LAKSHMI M

### Register Number: 212224040082

```python

print(f"Total number of test samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)


# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

# Train the model
# Write your code here
train_model(model,train_loader,test_loader)

test_model(model, test_loader)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="731" height="770" alt="Screenshot 2026-02-24 112648" src="https://github.com/user-attachments/assets/1b5651eb-6f63-4fef-96d3-112028e9db3e" />


## Confusion Matrix

<img width="678" height="657" alt="Screenshot 2026-02-24 112549" src="https://github.com/user-attachments/assets/87d7331e-5b99-478e-b1ee-40245c658c77" />


## Classification Report

<img width="589" height="258" alt="Screenshot 2026-02-24 112600" src="https://github.com/user-attachments/assets/94d37634-8761-470f-9d2a-7f292def6f1b" />


### New Sample Data Prediction

<img width="432" height="502" alt="Screenshot 2026-02-24 112510" src="https://github.com/user-attachments/assets/8aa09b9d-648e-4e11-83ac-c5883602b483" />

<img width="459" height="503" alt="Screenshot 2026-02-24 112459" src="https://github.com/user-attachments/assets/b589d6bc-eeb3-42a5-a93c-2568f4e9cd41" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
