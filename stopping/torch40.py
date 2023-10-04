import torch
from transformers import BertTokerizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-mltilingual-cased') 

text = "나는 파이토치를 이용한 딥러닝을 학습중이다."
marked_text= "[CLS]" + text + "[SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)

text = "과수원에 사과가 많았다."\
    "친구가 나에게 사과했가."\
        "백설공주는 독이 든 사과를 먹었다."
marked_text = "[CLS]" + text + "[SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(fokenized_text)
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'/format(tup[0], tup[1]))
    
segments_ids = [1] * len(tokenized_text)
print(segments_ids)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                  output_hidden_states=True)
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    
print("계층 개수:", len(hidden_states)," (initial embrdings + 12 BERT layers)")
layer_i = 0
print("배치 개수:", len(hidden_states[layer_i]))
batch_i = 0
print("토큰 개수:", len(hidden_states[layer_i][batch_i]))
token_i = 0
print("은닉층의 유닛 개수:", len(hidden_states[layer_i][batch_i][token_i]))

print('은닉 상태의 유형: ', )
