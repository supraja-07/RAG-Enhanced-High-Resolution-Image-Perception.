import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rap.vlm.base import BaseModel
from rap.trainer import RAPTrainer
import argparseimport pandas as pd
import base64
from io import BytesIO
from PIL import Image
import os

class RAPDataset(Dataset):
    def __init__(self, tsv_file):
        self.data = pd.read_csv(tsv_file, sep='\t')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Decode base64 image
        image_data = base64.b64decode(row['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        question = row['question']
        options = [row['A'], row['B'], row['C'], row['D']]
        
        # Convert answer to index
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        label = answer_map[row['answer']]
        
        return {
            'image': image,
            'question': question,
            'options': options,
            'label': label
        }
def train_rap(model_path='openbmb/VisRAG-Ret', epochs=10, lr=1e-4, batch_size=4, data_file='train_data.tsv'):
    # Initialize model
    model = BaseModel(rag_model_path=model_path)
    
    # Freeze everything except weight_predictor
    for param in model.parameters():
        param.requires_grad = False
    for param in model.weight_predictor.parameters():
        param.requires_grad = True
    
    # Trainer
    trainer = RAPTrainer(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.weight_predictor.parameters(), lr=lr)
    
    # Dataset and DataLoader
    dataset = RAPDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training on {len(dataset)} samples")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['image']
            questions = batch['question']
            options_list = batch['options']
            labels = batch['label']
            
            batch_loss = 0
            for i in range(len(images)):
                # Prepare message for model
                question_with_options = f"{questions[i]}\nOptions:\nA. {options_list[i][0]}\nB. {options_list[i][1]}\nC. {options_list[i][2]}\nD. {options_list[i][3]}"
                message = [
                    {'type': 'image', 'value': images[i]},
                    {'type': 'text', 'value': question_with_options}
                ]
                
                # Get model logits (this needs to be implemented in your MLLM subclass)
                try:
                    logits = model.get_logits_for_training(message, options_list[i])
                    
                    # Compute loss for this sample
                    loss = trainer.compute_loss(logits.unsqueeze(0), torch.tensor([labels[i]]))
                    batch_loss += loss
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Save the trained weight_predictor
    torch.save(model.weight_predictor.state_dict(), 'trained_weight_predictor.pth')
    print("Training complete. Saved to trained_weight_predictor.pth")
    
    # Save the trained weight_predictor
    torch.save(model.weight_predictor.state_dict(), 'trained_weight_predictor.pth')
    print("Training complete. Saved to trained_weight_predictor.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='openbmb/VisRAG-Ret')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_file', default='train_data.tsv', help='Path to training data TSV file')
    args = parser.parse_args()
    
    train_rap(args.model_path, args.epochs, args.lr, args.batch_size, args.data_file)