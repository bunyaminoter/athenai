import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

MODEL_PATH = "./saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    input_text = query.question + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Modelden cevap üretimi
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,  # daha kısa cevap
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.7,  # daha sıkı nucleus sampling
            temperature=0.6,  # daha düşük sıcaklık = daha tutarlı cevaplar
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # aynı 2 kelimelik grupların tekrarlanmasını engelle
            early_stopping=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # cevabı sadeleştirdm
    answer = response[len(query.question):].strip()

    return {"answer": answer}
