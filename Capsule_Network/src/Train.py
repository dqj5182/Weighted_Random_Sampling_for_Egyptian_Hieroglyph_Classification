import torch

num_classes = 40

# check if CUDA is available
TRAIN_ON_GPU = torch.cuda.is_available()

def train(capsule_net, criterion, optimizer, n_epochs, train_loader, print_every=300):
    '''Trains a capsule network and prints out training batch loss statistics.
       Saves model parameters if *validation* loss has decreased.
       param capsule_net: trained capsule network
       param criterion: capsule loss function
       param optimizer: optimizer for updating network weights
       param n_epochs: number of epochs to train for
       param print_every: batches to print and save training loss, default = 100
       return: list of recorded training losses
       '''

    # track training loss over time
    losses = []

    # one epoch = one pass over all training data
    for epoch in range(1, n_epochs + 1):

        # initialize training loss
        train_loss = 0.0

        capsule_net.train()  # set to train mode

        # get batches of training image data and targets
        for batch_i, (images, target) in enumerate(train_loader):
            # reshape and get target class
            target = torch.eye(num_classes).index_select(dim=0, index=target)

            if TRAIN_ON_GPU:
                images, target = images.cuda(), target.cuda()

            # zero out gradients
            optimizer.zero_grad()
            # get model outputs
            caps_output, reconstructions, y = capsule_net(images)
            # calculate loss
            loss = criterion(caps_output, target, images, reconstructions)
            # perform backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # accumulated training loss

            # print and record training stats
            if batch_i % 20 == 19:
                avg_train_loss = train_loss / print_every
                losses.append(avg_train_loss)
                print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch, avg_train_loss))
                train_loss = 0  # reset accumulated training loss

        return losses