import torch
from transformers import AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained("bergum/xtremedistil-emotion", torchscript=True)

model = model.eval()
model = model.to(device)

bs = 224
seq_len = 128
dummy_inputs = [
    torch.randint(1000, (bs, seq_len)).to(device),
    torch.ones(bs, seq_len, dtype=torch.int).to(device),
    torch.zeros(bs, seq_len, dtype=torch.int).to(device),
]

traced_model = torch.jit.trace(model, dummy_inputs)
torch.jit.save(traced_model, "model.pt")
