import torch

def evaluate(classifier, test_loader):
    def accuracy(outputs, labels):
        correct = 0
        total = 0
        for output, label in zip(outputs, labels):
            # print(output.argmax(), label)
            total += 1
            if label == output.argmax():
                correct += 1
        return correct, total

    correct = 0
    total = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            classifier.eval()
            outputs = classifier(inputs)
            # print(outputs, labels)
            correct_inc, total_inc = accuracy(outputs, labels)
            correct += correct_inc
            total += total_inc

    return correct / total
