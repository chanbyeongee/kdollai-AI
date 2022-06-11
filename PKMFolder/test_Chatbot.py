from AITransformer import transformer, predict
import pickle

# 하이퍼파라미터
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

with open('C:/MyProjects/Other Projects/AI ChatBot Project/Training_Model_Weights/Transformer_chatbot3/tokenizer.pickle', 'rb') as f:
  tokenizer = pickle.load(f) # 단 한줄씩 읽어옴

VOCAB_SIZE = tokenizer.vocab_size + 2
print(VOCAB_SIZE)

new_model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights\Transformer_chatbot3\Chatbot_Transformer3_weights")

output = predict("영화 볼래?", tokenizer, new_model)