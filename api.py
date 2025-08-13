import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from translate_client import Translator

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",  # React uygulamanın çalıştığı adres
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Buraya izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Google Cloud çeviri ayarları
GOOGLE_PROJECT_ID = "chatbot-translate-468818"  # kendi proje ID'niz
translator = Translator(GOOGLE_PROJECT_ID)


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
    user_text = query.question

    # 1. Dil algılama
    detected_lang = translator.detect_language(user_text)

    # 2. İngilizce değilse İngilizceye çevir
    if detected_lang != "en":
        user_text_en = translator.translate_text(user_text, "en")
    else:
        user_text_en = user_text

    # 3. Model input hazırlama
    input_text = user_text_en + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 4. Cevap üretimi
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.5,
            temperature=0.3,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    response_en = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer_en = response_en[len(user_text_en):].strip()


    # 5. Cevabı tekrar orijinal dile çevir
    if detected_lang != "en":
        answer = translator.translate_text(answer_en, detected_lang)
    else:
        answer = answer_en


    return {"answer": answer, "lang": detected_lang}
