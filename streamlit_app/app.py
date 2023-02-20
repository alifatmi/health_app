import streamlit as st
st.title("AI Medical Assistant")

# Get user input

text = st.text_input("Please Proved your text:")
print('ok')
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('./streamlit_app/savemodel')
model = GPT2LMHeadModel.from_pretrained('./streamlit_app/savemodel')

# Generate text
if len(text)>0:
  input_ids = tokenizer.encode(text, return_tensors='pt')
  output = model.generate(input_ids, max_length=20, do_sample=True,pad_token_id=tokenizer.eos_token_id)
#   model.generate(**encoded_input, pad_token_id=tokenizer.eos_token_id)

  st.write(tokenizer.decode(output[0], skip_special_tokens=True))
else:
  st.write('Welcome to GPT2')

# Display output
