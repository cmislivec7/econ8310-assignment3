import torch
from torch.utils.data import DataLoader
from assignment3 import ResNet18
import nnhelper as nnh 


# Specify the path
PATH = "model.pt"
# Create a new "blank" model to load our information into
model = ResNet18()
# Recreate the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Load back all of our data from the file
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']

# Uncomment below code if you want to continue training the model


#train_image_url = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz?raw=true'
#train_label_url = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-labels-idx1-ubyte.gz?raw=true'
#test_image_url =  'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-images-idx3-ubyte.gz?raw=true'
#test_label_url =  'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-labels-idx1-ubyte.gz?raw=true'


#train_data = nnh.CustomMNIST(train_image_url, train_label_url)
#test_data = nnh.CustomMNIST(test_image_url, test_label_url)


#train_dataloader = DataLoader(train_data, batch_size=64)
#test_dataloader = DataLoader(test_data, batch_size=64)

#model = model.to('cuda')  # Move model to the appropriate device

# Continue training for additional epochs
#additional_epochs = 10

# Make sure to update the epoch count correctly
#model.EPOCH = EPOCH


#nnh.train_net(model, train_dataloader, test_dataloader, epochs=additional_epochs, learning_rate=0.001, batch_size=64)

#Save model again after training for additional epochs

#EPOCH += additional_epochs  # Update the epoch count to reflect the new training period
#torch.save({
#    'epoch': EPOCH,
#    'model_state_dict': model.state_dict(),
#    'optimizer_state_dict': optimizer.state_dict(),
#}, PATH)
