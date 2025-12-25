import torch
import os
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

def train(model,train_dataloder,test_dataloader,loss_fn,optimizer,device,epochs,save_dir,log_dir,debug_mode=False):
    """_summary_

    Args:
        model (_type_): _description_
        train_dataloder (_type_): _description_
        test_dataloader (_type_): _description_
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        epochs (_type_): _description_
        save_dir (_type_): _description_
        log_dir (_type_): _description_
        debug_mode (bool, optional): _description_. Defaults to False.
    """
    
    model.train()
    model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,1e-6)
    scaler = torch.amp.GradScaler(device)
    
    writer = SummaryWriter(log_dir)
    
    for epoch in range(epochs):
        
        # ===================== Training =====================
        
        train_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        for x,y in tqdm(train_dataloder,desc=f'[Epoch {epoch}/{epochs}]Training...'):
            x, y = x.to(device), y.to(device)
            
            logits = model(x) # (BS,seq_len) -> (BS,seq_len,vocab_size)
            out = logits.view(-1,logits.size(-1)) # (BS,seq_len,vocab_size) -> (BS * seq_len,vocab_size)
            flat_y = y.view(-1) # (BS,seq_len) -> (BS * seq_len)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device):
                loss = loss_fn(out,flat_y)
                train_loss += loss.item()
            
            scaler.scale(loss).backward()
            
            with torch.no_grad():
                preds = torch.argmax(out,dim=-1)
                total_correct += (preds == flat_y).sum().item()
                total_tokens += flat_y.numel()
                
                if debug_mode:
                    rand_int = random.randint(0,2000)
                    if rand_int >= 1900:
                        print("===================== DEBUG INFO =====================\n")
                        print(f"Target:{flat_y[0:30]}\n")
                        print(f"Prediction:{preds[0:30]}\n")                     
                        print(f"Grad norm:{grad_norm(model)}\n")                     
                        
            optimizer.step()
            
        mean_train_loss = train_loss / (len(train_dataloder) + 1e-6)
        train_ppx = torch.exp(torch.tensor(mean_train_loss))
        train_accuracy = total_correct / (total_tokens + 1e-6) * 100
        
        print(f'Epoch:{epoch + 1} | Train loss:{mean_train_loss:.4f} | Train perplexity:{train_ppx:.4f} | Train accuracy:{train_accuracy:.2f} %')
        
        
        
        # ===================== Evaluating =====================
        
        test_loss = 0.0
        test_total_tokens = 0
        test_total_correct = 0
        
        for x,y in tqdm(test_dataloader,desc='Evaluating...'):
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                logits = model(x) # (BS,MAX_SEQ_LEN,VOCAB_SIZE)
                out = logits.view(-1,logits.size(-1)) # (BS,MAX_SEQ_LEN,VOCAB_SIZE) -> (BS * MAX_SEQ_LEN,VOCAB_SIZE)
                flat_y = y.view(-1) # (BS,MAX_SEQ_LEN) ->  (BS * MAX_SEQ_LEN)
                loss = loss_fn(out,flat_y)
                
                test_loss += loss.item()
                
                preds = torch.argmax(out,dim=-1) # (BS * MAX_SEQ_LEN,VOCAB_SIZE) -> (BS * MAX_SEQ_LEN)
                test_total_correct += (preds == flat_y).sum().item()
                test_total_tokens += y.numel()
        
        mean_test_loss = test_loss / len(test_dataloader)
        test_ppx = torch.exp(torch.tensor(mean_test_loss))
        test_accuracy = test_total_correct / (test_total_tokens +  1e-6) * 100
        
        print(f'Epoch:{epoch + 1} | Test loss:{mean_test_loss:.4f} | Test perplexity:{test_ppx:.4f} | Test accuracy:{test_accuracy:.2f} %')
        
        if (epoch + 1) == 5:
            torch.save(model,os.path.join(save_dir,fr'tinyGPT_{epoch + 1}epochs.pth'))
        
        writer.add_scalar('Loss/Train',train_loss,epoch + 1)
        writer.add_scalar('Loss/Test',test_loss,epoch + 1)
        writer.add_scalar('Accuracy/Train',train_accuracy,epoch + 1)
        writer.add_scalar('Accuracy/Test',test_accuracy,epoch + 1)
        writer.add_scalar('Perplexity/Train',train_ppx,epoch + 1)
        writer.add_scalar('Perplexity/Test',test_ppx,epoch + 1)

def grad_norm(model):
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    
    return total_norm ** 0.5
            