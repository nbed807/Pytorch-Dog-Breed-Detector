from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def main():
    #################### MODEL Training ####################
    # Ask the user if they would like to train the model
    train = input("Would you like to first train the model? [y/n] ")
    if train in ['y', 'Y', 'yes', 'Yes', 'YES']:

        # Image data directory in ImageFolder structure
        data_dir = "images"

        # Models to choose from (alexnet, inception, squeezenet, vgg) - Inception by default as it showed the best results
        model_name = input("Please enter one the the listed models you'd like to train ('alexnet', 'inception', 'squeezenet', 'vgg'): ")

        # Number of classes in the dataset
        num_classes = 120

        # Batch size for training
        batch_size = 8

        # Number of epochs to train for
        num_epochs = 15

        # Choose to feature extract or to finetune whole model
        feature_extract = True

        # Prompt to let the user know the program is still running, Inception takes ~3min to load in
        if model_name == "inception":
            print("Please be patient; Inception takes a while to load if scipy version newer than 1.3.3")

        # Defining the model training function
        def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
            since = time.time()

            val_acc_history = []

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if is_inception and phase == 'train':
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        val_acc_history.append(epoch_acc)

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history



        # Set for feature extracting
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False



        def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
            # Initialize these variables which will be set in this if statement. Each of these
            #   variables is model specific.
            model_ft = None
            input_size = 0

            if model_name == "alexnet":
                model_ft = models.alexnet(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224

            elif model_name == "vgg":
                model_ft = models.vgg16_bn(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224

            elif model_name == "squeezenet":
                model_ft = models.squeezenet1_0(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                model_ft.num_classes = num_classes
                input_size = 224

            elif model_name == "inception":
                model_ft = models.inception_v3(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                # Handle the auxilary net
                num_ftrs = model_ft.AuxLogits.fc.in_features
                model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
                # Handle the primary net
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs,num_classes)
                input_size = 299 # Inception requires 299px image size

            else:
                print("Invalid model name, exiting...")
                exit()

            return model_ft, input_size

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained = feature_extract)


        # Data normalisation for both training and validation - 
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Model", model_name, "is being trained with", device)

        # Send the model to the available device
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(model_ft,
                                     dataloaders_dict,
                                     criterion, optimizer_ft,
                                     num_epochs = num_epochs,
                                     is_inception = (model_name=="inception"))

        # Plot the training curves of validation accuracy vs. number of training epochs
        ohist = []

        ohist = [h.cpu().numpy() for h in hist]

        plt.title("Validation Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1,num_epochs+1),ohist,label=model_name)
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, num_epochs+1, 1.0))
        plt.legend()
        plt.show()



        # Save the trained model to models folder
        torch.save(model_ft, "models/" + model_name + "_scratch.pt")
    


    #################### MODEL PREDICTION ####################
    # Define and set up transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        )
        ])

    # Import Pillow and open image
    from PIL import Image

    breed = input("Please enter the name of the '.jpg' you would like to check: ")
    img = Image.open(breed + ".jpg")

    # Transform Image
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)


    # Load and model inference
    model_t = input("Please enter the name of the model you trained: ")
    if model_t not in ['alexnet', 'inception', 'squeezenet', 'vgg']:
        print("Invalid model name, exiting...")
        exit()

    trained_model = torch.load("models/" + model_t + ".pt")

    # Run the GPU trained model with the CPU
    trained_model = trained_model.cpu().float()
    trained_model.eval
    output = trained_model(batch_t)

    # Uncheck to check output tensor dimension
    # print(output.shape)

    # Read Labels
    with open('breeds.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(output, 1)
    percentage = torch.nn.functional.softmax(output, dim = 1)[0] * 100

    # Show the user the prediction
    print("We think that this dog is a ", classes[index[0]], ". (",round(percentage[index[0]].item(),2),"% confidence)", sep="")

    # If the prediction isn't right, the user could see the top five results
    txt = input("Would you like to see the other predictions? [y/n] ")

    if txt in ['y', 'Y', 'yes', 'Yes', 'YES']:

        _, indices = torch.sort(output, descending=True)
        list = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

        for item, percent in list:
            print(item, " ", round(percent,2), "%", sep="")


if __name__ == "__main__":
    main()

