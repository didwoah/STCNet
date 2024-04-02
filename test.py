import torch
from torch.utils.data import DataLoader
from dataset import Nina1Dataset, Nina2Dataset  # Update this according to your dataset classes
from networks.EMGHandNet import EMGHandNetCE  # Update this with your model class
from util import AccuracyMeter, get_data
import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser(description='Argument for model evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--dataset', type=str, required=True, choices=['nina1', 'nina2', 'nina4'], help='Dataset type for evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    parser.add_argument('--sampled', action='store_true', help='Evaluate on a sampled dataset')
    return parser.parse_args()

def load_model(model_path, dataset, sampled):

    model = EMGHandNetCE(data=f"{dataset}{'_sampled' if sampled else ''}")
    model.load_state_dict(torch.load(model_path)['model'])
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def evaluate_model(model, test_loader):
    model.eval()
    accuracy_meter = AccuracyMeter()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = model(inputs)
            accuracy_meter.update(outputs, labels)
    
    return accuracy_meter.correct / accuracy_meter.total

def main():
    opt = parse_option()

    _, test = get_data(opt.dataset, -1)

    if opt.dataset == 'nina1':
        test_dataset = Nina1Dataset(test, mode='labels', model='EMGHandNet')
    elif opt.dataset in ['nina2', 'nina4']:
        test_dataset = Nina2Dataset(test, sampled=opt.sampled, mode='labels', model='EMGHandNet')
    else:
        raise ValueError(f"Unsupported dataset type: {opt.dataset}")

    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    model = load_model(opt.model_path, opt.dataset, opt.sampled)
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
