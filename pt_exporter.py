import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", torchscript=True)

model = model.eval()
model = model.to(device)

bs = 224
seq_len = 128
dummy_inputs = [
    torch.randint(1000, (bs, seq_len)).to(device),
    torch.zeros(bs, seq_len, dtype=torch.int).to(device),
]


traced_model = torch.jit.trace(model, dummy_inputs)
torch.jit.save(traced_model, "model.pt")
