from kobert_transformers import get_tokenizer, get_kobert_model
import torch
from transformers import BertModel, BertTokenizer

from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("monologg/kobert")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")


text = "안녕하세요. KoBERT를 사용하여 텍스트를 처리하고 있습니다."
encoded_input = tokenizer.encode_plus(text, return_tensors='pt')
output = model(**encoded_input)
print(output.last_hidden_state.shape)  # Should output: (batch_size, num_tokens, 768)
