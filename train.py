#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from model import GPTLanguageModel, block_size

# Настройки обучения
batch_size = 64 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Фиксируем seed для воспроизводимости
torch.manual_seed(1337)

print("Загрузка датасета...")
with open('idiot.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level токенизация
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Размер словаря (уникальных символов): {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# Train / Val сплит (90% на 10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Инициализация модели на {device}...")
model = GPTLanguageModel(vocab_size=vocab_size)
m = model.to(device)

print(f"Параметров в модели: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Старт обучения...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Шаг {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Обучение завершено. Сохранение весов...")
torch.save(m.state_dict(), 'dostoevsky_model.pt')

print("Генерация тестового текста:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)

