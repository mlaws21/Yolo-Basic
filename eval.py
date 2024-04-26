import torch

def evaluate(classifer, test_loader):

    def accuracy(outputs, labels):
        correct = 0
        total = 0
        for (output, one_hot_label) in zip(outputs, labels):
            _, label = torch.max(one_hot_label, dim=0)
            total += 1
            # print(label)
            if label == output.argmax():
                correct += 1
        return correct, total
    correct = 0
    total = 0
    for data in test_loader:    
    # for data in loader(partition):
        inputs, labels = data
        outputs = classifer(inputs) 
        correct_inc, total_inc = accuracy(outputs, labels)
        correct += correct_inc
        total += total_inc
    return correct / total