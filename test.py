import torch
from torch.utils.data import DataLoader
from dataset import NinaDataset  # Update this according to your dataset classes
from util import AccuracyMeter, MetricsMeter, get_data, get_model
import argparse
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser(description='Argument for model evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--dataset', type=str, required=True, choices=['nina1', 'nina2', 'nina4'], help='Dataset type for evaluation')
    parser.add_argument('--model', type=str, default='STCNet', choices=['baseline', 'STCNet'], help='Model type for evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    return parser.parse_args()

def load_model(opt):
    model = get_model(opt)
    model.load_state_dict(torch.load(opt.model_path)['model'])
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def evaluate_model(model, test_loader, dataset):
    model.eval()
    metrics_meter = MetricsMeter(dataset)
    acc_meter = AccuracyMeter()
    
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = model(inputs)
            metrics_meter.update(outputs, labels)
            acc_meter.update(outputs, labels)
 
    return acc_meter.compute(), metrics_meter.compute_metrics()

def evaluate_by_subjects(model, test_loader):
    acc_by_subjects = {}
    model.eval()
    with torch.no_grad():
        for inputs, labels, subjects in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Model prediction
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Aggregate accuracy for each subject
            for i, subject in enumerate(subjects):
                subject = subject.item()  # Process subject as string or number
                if subject not in acc_by_subjects:
                    acc_by_subjects[subject] = {'correct': 0, 'total': 0}

                # Check if the prediction is correct
                if preds[i] == labels[i]:
                    acc_by_subjects[subject]['correct'] += 1
                acc_by_subjects[subject]['total'] += 1

    # Calculate accuracy for each subject
    for subject in acc_by_subjects:
        correct = acc_by_subjects[subject]['correct']
        total = acc_by_subjects[subject]['total']
        acc_by_subjects[subject] = correct / total * 100 if total > 0 else 0.0

    return acc_by_subjects

def main():
    opt = parse_option()

    _, test = get_data(opt.dataset, -1)

    test_dataset = NinaDataset(test, dataset=opt.dataset, model=opt.model)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    model = load_model(opt)
    metrics, metrics_details = evaluate_model(model, test_loader, opt.dataset)
    
    print("Accuracy")
    print(metrics)
    print("Metrics")
    print(metrics_details)
    
    acc_by_subjects = evaluate_by_subjects(model, test_loader)
    print("Accuracy across subjects")
    print(acc_by_subjects)
    print("inter-subject score")
    print(np.std(list(acc_by_subjects.values())))

if __name__ == '__main__':
    main()
