from transformers import BertTokenizer, BertModel, AutoModelForMaskedLM
tokenizer = BertTokenizer.from_pretrained(r'chinese_roberta_L-8_H-512')
model = AutoModelForMaskedLM.from_pretrained(r'chinese_roberta_L-8_H-512')
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
