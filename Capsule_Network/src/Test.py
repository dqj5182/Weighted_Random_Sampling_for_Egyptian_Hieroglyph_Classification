import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Capsule_Loss import CapsuleLoss

num_classes = 40

# check if CUDA is available
TRAIN_ON_GPU = torch.cuda.is_available()

criterion = CapsuleLoss()

def test(capsule_net, test_loader, classes):
    '''Prints out test statistics for a given capsule net.
       param capsule_net: trained capsule network
       param test_loader: test dataloader
       return: returns last batch of test image data and corresponding reconstructions
       '''
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    test_loss = 0  # loss tracking

    labels = []
    predictions = []

    capsule_net.eval()  # eval mode

    for batch_i, (images, target) in enumerate(test_loader):
        target = torch.eye(num_classes).index_select(dim=0, index=target)

        batch_size = images.size(0)

        if TRAIN_ON_GPU:
            images, target = images.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        caps_output, reconstructions, y = capsule_net(images)
        # calculate the loss
        loss = criterion(caps_output, target, images, reconstructions)
        # update average test loss
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(y.data.cpu(), 1)
        _, target_shape = torch.max(target.data.cpu(), 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target_shape.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target_shape.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        # Will be used for calculating Recall, Precision, and F1-score
        labels.extend(target_shape.data.view_as(pred).tolist())
        predictions.extend(pred.tolist())

        # avg test loss
    avg_test_loss = test_loss / len(test_loader)
    print('Test Loss: {:.8f}\n'.format(avg_test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    # Total Test accuracy
    print("\nAccuracy: {:.3%}".format(accuracy_score(labels, predictions)))
    print("\nPrecision: {:.3%}".format(precision_score(labels, predictions, average='weighted')))
    print("\nRecall: {:.3%}".format(recall_score(labels, predictions, average='weighted')))
    print("\nF1-score: {:.3%}".format(f1_score(labels, predictions, average='weighted')))

    # return last batch of capsule vectors, images, reconstructions
    return caps_output, images, reconstructions