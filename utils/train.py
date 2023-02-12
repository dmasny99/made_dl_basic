import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from utils.utils import decode_nums
from IPython.display import clear_output
import matplotlib.pyplot as plt


def train(model, 
          optimizer,  
          criterion,
          metric,
          epoch_num,
          train_loader, 
          test_loader, 
          inv_char_dict, 
          device):
    
    min_test_cer = 1
    train_loss = []
    train_cer = []
    
    test_loss = []
    test_cer = []
    for epoch_number in range(epoch_num):
        model.train()


        batch_cer = 0
        batch_loss = 0
        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            
            out, input_lengths, target_lengths = model(img)
            log_prob = F.log_softmax(out, dim=2)
            
            loss = criterion(log_prob, labels, input_lengths, target_lengths)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # computing CER
            out = out.permute(1, 0, 2)
            out = out.log_softmax(2)
            out = out.argmax(-1)
            decoded_out = decode_nums(out, inv_char_dict)
            decoded_labels = decode_nums(labels, inv_char_dict)
            
            cer = metric(decoded_out, decoded_labels)
            batch_cer += cer
            
    
        train_loss.append(batch_loss / len(train_loader))
        train_cer.append(batch_cer / len(train_loader))
        
        model.eval()
        batch_cer = 0
        batch_loss = 0
        
        with torch.no_grad():
            for img, labels in test_loader:
                img, labels = img.to(device), labels.to(device)
                out, input_lengths, target_lengths = model(img)
                log_prob = F.log_softmax(out, dim=2)
                loss = criterion(log_prob, labels, input_lengths, target_lengths)
                batch_loss += loss.item()

                # computing CER
                out = out.permute(1, 0, 2)
                out = out.log_softmax(2)
                out = out.argmax(-1)
                decoded_out = decode_nums(out, inv_char_dict)
                decoded_labels = decode_nums(labels, inv_char_dict)

                cer = metric(decoded_out, decoded_labels)
                batch_cer += cer
                
            test_loss.append(batch_loss / len(test_loader))
            test_cer.append(batch_cer / len(test_loader))
            
            if test_cer[-1] < min_test_cer:
                torch.save(model.state_dict(), "./ocr_model.pth")
        
        clear_output(wait=True)
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        
        ax[0].plot(train_loss, "b", label="train")
        ax[0].plot(test_loss, "orange", label="test")
        ax[0].set_title("CTC Loss")
        ax[0].set_xlabel("Epoch num", fontsize=12)
        ax[0].set_ylabel("CTC Loss", fontsize=12)
        ax[0].legend(loc="lower right")
        ax[0].grid(True)
        
        ax[1].plot(train_cer, "b", label="train")
        ax[1].plot(test_cer, "orange", label="test")
        ax[1].set_title("Character Error Rate")
        ax[1].set_xlabel("Epoch num", fontsize=12)
        ax[1].set_ylabel("CER", fontsize=12)
        ax[1].legend(loc="lower right")
        ax[1].grid(True)
        
        plt.show();
        print(f"Training CER: {train_cer[-1]}   Test CER: {test_cer[-1]}")    
        