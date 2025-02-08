import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as model_list
import matplotlib.pyplot as plt
import datetime
import argparse
from torchvision.datasets import VOCSegmentation
from torchvision.datasets import CIFAR100,CIFAR10,MNIST

from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from PIL import Image

def train_model(model,n_epochs,train_loader,device,loss_fn,optimizer,scheduler,save_file,plot_file,val_loader,log_file,save_log):
    # inputs : model, n_epochs, loss_fn, optimizer, scheduler, train_loader,val_loader, loss_plot,log_file, model_file, devilce
    model.train()
    losses_train =[]
    val_losses_train = []
    for epoch in range(1,n_epochs):
        model.train()
        loss_train =0.0
        for data in train_loader:
            imgs =data[0]
            imgs = imgs.to(device=device)
            train_label = data[1]
            train_label = train_label.to(device=device)
            train_label = train_label.type(torch.long)
            imgs = imgs.type(torch.float32)
            outputs=model(imgs)
            print("Here\n")
            print(train_label)
            classification_labels = torch.mode(train_label.view(batch_size, -1), dim=1).values

            print("There\n")
            loss = loss_fn(outputs,train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]
        log_train_msg = '{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader))
        print(log_train_msg)
        if (save_log == 1):
            with open(log_file, "a") as file:
                file.write(log_train_msg)
                file.write("\n")

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradient tracking during validation
            for val_data in val_loader:
                val_img = val_data[0]
                val_img = val_img.to(device=device)
                val_img = val_img.type(torch.float32)
                val_label = val_data[1]
                val_label = val_label.to(device=device)
                val_label = val_label.type(torch.long)
                val_outputs = model(val_img)
                print("Here\n")
                print(val_label.squeeze(1))
                print("There\n")
                loss = loss_fn(val_outputs, val_label)
                val_loss += loss.item()  # Accumulate validation loss

        # Calculate average validation loss
        val_losses_train += [val_loss / len(val_loader)]

        log_val_msg = '{} Epoch {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, val_loss / len(train_loader))

        print(log_val_msg)
        if (save_log == 1):
            with open(log_file, "a") as file:
                file.write(log_val_msg)
                file.write("\n")

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(val_losses_train, label='Val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)


def main():
    global  save_file, n_epochs, batch_size
    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='save file', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argParser.add_argument('-lr', metavar='learning rate', type=float, help='What is the learning rate')
    argParser.add_argument('-lf', metavar='loss function', type=int, help='What is the loss function')
    argParser.add_argument('-sf', metavar='scheduler function', type=int, help='What is the scheduler function')
    argParser.add_argument('-of', metavar='optimizer function', type=int, help='What is the optimizer function')
    argParser.add_argument('-sl', metavar='save log', type=int, help='Do you want to save the log')
    argParser.add_argument('-sp', metavar='save plot', type=int, help='Do you want to save the loss plot')

    # inputs 
    # m : model path 
    # e number of epochs
    # b : batchsize
    # p : plot path 
    # lr : learning rate
    # lf : loss function
    # sf : scheduler function
    # of : optimizer function
    # sp : save plot 
    # sl : save log 

    args = argParser.parse_args()

    if args.m != None:
        save_file = args.m
    else:
        save_file = "model.pth"
    if args.e != None:
        n_epochs = args.e
    else:
        n_epochs = 50
    if args.b != None:
        batch_size = args.b
    else:
        batch_size =1
    if args.p != None:
        plot_file = args.p
    else:
        plot_file = "loss.png"
    if args.lr != None:
        learning_rate = float(args.lr)
    else:
        learning_rate = 1e-3
    if args.lf != None:
        loss_func = args.lf
    else:
        loss_func = 0
    if args.of != None:
        optimizer_func = args.of
    else:
        optimizer_func = 0
    if args.sf != None:
        scheduler_func = args.sf
    else:
        scheduler_func = 0

    if args.sl != None:
        save_log = args.sl
    else:
        save_log =0
    if args.sp != None:
        save_plot = args.sp
    else:
        save_plot =0

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)
    print('\t\tlearning rate = ', learning_rate)
    print('\t\tloss function = ', loss_func)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)


## LOADING MODEL
    # replace the model with the model you are using 
    #model = model_list.mobilenet_v3_small(pretrained = True) # change this line with your own model (create an instance of your own model)
    model = model_list.resnet18(pretrained =True)

    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the final fully connected layer to output 10 classes (for MNIST)
    model.fc = nn.Linear(512, 10)

    model.to(device)
    model.apply(init_weights)



    # SETTING UP OPTIMIZER< SCHEDULER AND LOSS FUNCTION
    if(optimizer_func==0):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if(scheduler_func == 0):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_func_name = ""
    if(loss_func == 0):
        loss_fn = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean',ignore_index=255)
        loss_func_name = "Cross Entropy"
    elif(loss_func ==1):
        # weighted cross entropy
        class_weights = get_class_weights(num_classes=21,data_loader=train_loader)
        class_weights = torch.tensor(class_weights,dtype = torch.float32)
        class_weights = class_weights.to(device=device)
        loss_fn = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean',weight=class_weights,ignore_index=255)
        loss_func_name = "Weighted Cross Entropy"
    elif(loss_func == 2):
        # DICE
        loss_fn = dice_loss
        loss_func_name = "DICE"




    # Transformation for Images
    img_transforms = transforms.Compose([
        transforms.Resize((512, 512)),   # Resize to match model input
        transforms.RandomHorizontalFlip(p=0.5),  # Augmentations
        transforms.RandomRotation(10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Transformation for Masks (segmentation labels)
    label_transforms = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to match image
        transforms.ToTensor()  # Converts to tensor but does NOT normalize
    ])
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
    ])
    # Define transform: Resize to 224x224 (ResNet expects this input size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize MNIST images
    ])

## SET UP DATASET AND DATALOADER
    #train_set = get_dataset('data/pascalvoc',"2012","train",img_transforms,label_transforms)
    #train_set = CIFAR100('data/cifar100', train=True, download=True, transform=train_transform)
    train_set = MNIST(root='./data', train=True, transform=transform, download=True)

    #VOCSegmentation('data/pascalvoc',year = "2012", image_set="train", download=True, transform=img_transforms,target_transform=label_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #val_set = VOCSegmentation('data/pascalvoc',year = "2012", image_set="val", download=True, transform=img_transforms,target_transform=label_transforms)
    #val_set = get_dataset('data/pascalvoc', "2012", "val", img_transforms, label_transforms)
    #val_set = CIFAR100('data/cifar100', train=False, download=True, transform=train_transform)    
    val_set = MNIST(root='./data', train=False, transform=transform, download=True)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if(save_log):
        parts = save_file.split('.')
        log_path = parts[0] + "_log.txt"
        create_log(log_path,save_file,plot_file,device,n_epochs,batch_size,learning_rate,loss_func,optimizer_func,scheduler_func)
    else:
        log_path = ""

    train_model(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
            save_file=save_file,
            plot_file = plot_file,
            val_loader = val_loader,
            log_file = log_path,
            save_log = save_log,
            )

###################################################################

# loss functions

# dice
def dice_loss(predictions,labels,num_classes=21):
    dice = []
    #print(predictions.requires_grad)
    #if(predictions.requires_grad == False):
    #    print("predictions required_grad is false")

    predictions = torch.softmax(predictions, dim=1)
    for class_idx in range(num_classes):
        pred_idx = predictions[:, class_idx, :, :]
        target_idx = (labels == class_idx).float()
        intersection = torch.sum(pred_idx*target_idx)
        union = torch.sum(pred_idx) + torch.sum(target_idx)
        if (union == 0):
            dice.append(torch.tensor(0.0,device=predictions.device,requires_grad=True))
        else:
            dice.append(2 * intersection / union)
    total_dice = torch.sum(torch.stack(dice))
    #if(total_dice.requires_grad == False ):
    #    print("total_dice grad == 0")
    #    print(total_dice)
    return 1 - total_dice

def weighted_dice_loss(predictions,labels,num_classes=21):
    dice = []
    predictions = torch.argmax(predictions, dim=1)
    for class_idx in range(num_classes):
        pred_idx = (predictions == class_idx)
        target_idx = (labels == class_idx)
        intersection = np.logical_and(pred_idx, target_idx).sum()
        union = np.logical_or(pred_idx, target_idx).sum()
        if (union == 0):
            dice.append(0)
        else:
            dice.append(2 * intersection / union)
    total_dice = torch.mean(torch.tensor(dice))
    return 1 - total_dice


# weighted loss entropy

def get_class_weights(num_classes,data_loader):
    class_pixel_count = np.zeros(num_classes)
    for data in data_loader:
        labels = data[1]
        batch_size= labels.shape[0]
        for idx in range(0,batch_size):
            batch_label = labels[idx]
            for class_id in range(num_classes):
                logic_tensor = batch_label == class_id
                class_pixel_count[class_id] = class_pixel_count[class_id] + np.sum(logic_tensor.numpy())
    total_pixels = class_pixel_count.sum()
    class_weights = total_pixels/(num_classes*class_pixel_count)
    class_weights /= class_weights.sum()
    return class_weights



def get_dataset(data_dir,year,image_set,img_transforms,label_transforms):
    file_year = "VOC"+year
    if(not os.path.exists(os.path.join(data_dir,"VOCdevkit",file_year))):
        data_set = VOCSegmentation(data_dir, year=year, image_set=image_set, download=True,
                                    transform=img_transforms, target_transform=label_transforms)
    else:
        data_set = VOCSegmentation(data_dir, year=year, image_set=image_set, download=False,
                                   transform=img_transforms, target_transform=label_transforms)
    return data_set

def init_weights(m):
    # MIGHT NEED TO CHANGE THE WEIGHT INTIALIZATION TO BE MORE GENERAL
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "fill") and m.bias is not None:
            m.bias.fill_(0.01)        
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, "fill") and m.bias is not None:
            m.bias.fill_(0.01)

def create_log(log_file,**kwargs):
    param = kwargs.items()    
    if(not os.path.exists(log_file)):
        mode = "a"
    else :
        mode = "w"
    with open(log_file,mode) as file:
        file.write("Parameters:\n")
        for key,valuje in kwargs.items():
            file.write(f"{key} : {value}\n")           
        file.write(f"Start training : {datetime.datetime.now()}\n")

def append_log(save_log,log_file,log_msg):
    if (save_log == 1):
        with open(log_file, "a") as file:
            file.write(log_msg)
            file.write("\n")


if __name__ == '__main__':
    main()
