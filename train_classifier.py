import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
# import clip
import numpy as np
from sklearn.model_selection import train_test_split
# from utils import CustomImageFolder
import torchvision.models as models
from argparse import ArgumentParser
from tqdm import tqdm
import random
import glob
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.utils import save_image
from utils.binary_classifier import simplenet
import torchvision


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, data_cnt=-1, y=0):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*/*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.jpg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.png")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        random.seed(17)
        random.shuffle(self.filenames)
        if data_cnt != -1:
            self.filenames = self.filenames[:data_cnt]
        if data_dir[-1] != '/':
            data_dir += '/'
        self.img_ids = [x.replace(data_dir, '').replace('/', '_') for x in self.filenames]
        self.transform = transform
        self.y = y

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y, filename

    def __len__(self):
        return len(self.filenames)

class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)


def load_dataset(args):
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), AddGaussianNoise(0., 0.4)])

    dataset_wm = CustomImageFolder(args.wm_dir,
                                       transform=transform, y=1, data_cnt=args.data_cnt)
    dataset_org = CustomImageFolder(args.org_dir,
                                        transform=transform, y=0, data_cnt=args.data_cnt)
 
    print(f'number of watermarked images --> {len(dataset_wm)}')
    print(f'number of original images --> {len(dataset_org)}')

    dataset = torch.utils.data.ConcatDataset([dataset_org, dataset_wm])
    train, test = train_test_split(np.arange(len(dataset_org) + len(dataset_wm)), test_size=0.2)

    train_set = torch.utils.data.Subset(dataset, train)
    test_set = torch.utils.data.Subset(dataset, test)

    print(f'train/test split: {len(train_set), len(test_set)}')

    return train_set, test_set


batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a binary classification model using ResNet18
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))


# Function to train the model
def train_model(args, model, train_dataloader, val_dataloader, num_epochs=50, lr=0.001):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # epsilon = 0.01
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels, _ in tqdm(train_dataloader):
            labels = labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            adv_inputs = inputs.detach()

            optimizer.zero_grad()
            outputs = model(adv_inputs).to(device)[:, 0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = torch.round(outputs)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        total_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        # Evaluation on the validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_dataloader):
                labels = labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)[:, 0]
                val_loss += criterion(outputs, labels).item()
                predicted = torch.round(outputs)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        torch.save({'model': model.state_dict()}, args.out_dir)

        accuracy = 100 * correct / total
        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")



# Main function to run the binary classification task
def main():
    parser = ArgumentParser()
    parser.add_argument("--wm-dir", type=str, required=True)
    parser.add_argument("--org-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--data-cnt", default=10000, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()

    print('loading dataset ...')
    train_set, test_set = load_dataset(args)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)  # check
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)  # check

    # Create the model and train it
    model = ResNetBinaryClassifier()
    train_model(args, model, train_loader, val_loader, num_epochs=args.epochs)
    

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
