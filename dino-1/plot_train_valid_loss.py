import json
import matplotlib.pyplot as plt

# Step 1: Read and Parse the JSON Data
with open('/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/DINO-ViTs Other TRAINED CHECKPOINTS/log.txt', 'r') as file:
    lines = file.readlines()

epochs = []
train_loss = []
val_loss = []

for line in lines:
    data = json.loads(line)
    epochs.append(data['epoch'])
    train_loss.append(data['train_loss'])
    val_loss.append(data['val_loss'])

# Step 2: Plot the Data
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')

# Adding titles and labels
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Show the plot
plt.savefig('training_validation_loss.png')
