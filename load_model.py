import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

model_checkpoint = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
device = torch.device('cpu') 

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state'][:, 0, :]
        x = self.classifier(x)
        return x

model = BertClassifier(num_labels=3)
model.load_state_dict(torch.load('bert_sa.pt', map_location=torch.device('cpu')))
#model.eval()

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

def predict_sentiment(phrase):
    # Tokenize input phrases
    encodings = tokenizer(phrase, truncation=True, padding=True, return_tensors='pt')

    # Move tensors to the device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, 1)

    # Convert predictions to sentiment labels
    predicted_labels = predictions.cpu().numpy()
    return [label_map[label] for label in predicted_labels][0]

#sentiment = predict_sentiment(["Curry for dinner."])
#print(sentiment[0])