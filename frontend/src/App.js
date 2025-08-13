import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Merhaba! 🎤 Başlat’a basıp konuşmaya başlayabilirsin." },
  ]);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const recognitionRef = useRef(null);
  const chatEndRef = useRef(null);

  // Ekranı her yeni mesajda en alta kaydır
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  // --- TTS (Bot cevabını seslendirme) ---
  const speak = (msg, lang) => {
    if (!msg) return;

    // Bot konuşurken mikrofona geri besleme olmasın
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {}
    }

    const utterance = new SpeechSynthesisUtterance(msg);

    // Dil güvenli eşleştirme
    const code = (lang || "").toLowerCase();
    let speechLang = "en-US";
    if (code.startsWith("tr")) speechLang = "tr-TR";
    else if (code.startsWith("en")) speechLang = "en-US";
    utterance.lang = speechLang;

    speechSynthesis.cancel(); // önceki konuşmaları iptal et
    speechSynthesis.speak(utterance);
  };

  // --- Yazılı mesaj gönder ---
  const handleSend = async (customText, fromVoice = false) => {
    const text = (customText ?? input).trim();
    if (!text) return;

    // kullanıcı mesajını ekle
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setIsThinking(true);

    try {
      const res = await axios.post("http://localhost:8000/chat", { question: text });
      const answer = res?.data?.answer ?? "(cevap alınamadı)";
      const lang = res?.data?.lang;

      // bot mesajını ekle
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);

      // sadece sesli konuşmadan geldiyse TTS
      if (fromVoice) speak(answer, lang);

    } catch (err) {
      console.error("API error:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Üzgünüm, bir hata oluştu. (API)" },
      ]);
    } finally {
      setIsThinking(false);
    }
};

  // --- Sürekli dinleme başlat ---
  const startListening = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      alert("Tarayıcınız ses tanımayı desteklemiyor.");
      return;
    }

    // Zaten dinliyorsak tekrar başlatma
    if (recognitionRef.current) {
      console.warn("Dinleme zaten açık.");
      return;
    }

    const rec = new SR();
    rec.lang = "tr-TR";          // başlangıç dili (backend’e göre cevap seslendiriliyor)
    rec.continuous = true;        // sürekli dinleme
    rec.interimResults = false;   // ara sonuç istemiyoruz

    // onresult içinden:
    rec.onresult = (e) => {
    const transcript = e.results[e.results.length - 1][0].transcript.trim();
    if (!transcript) return;

    // sadece textarea’yı güncelle (mesaj ekleme handleSend’da olacak)
    setInput(transcript);

    // Bot konuşurken mikrofona geri besleme olmasın
    try { rec.stop(); } catch {}

    // API çağrısı, TTS çalışsın (fromVoice=true)
    handleSend(transcript, true);
};


    rec.onerror = (e) => {
      console.error("SpeechRecognition error:", e.error);
    };

    rec.onend = () => {
      // Manuel akış: otomatik tekrar başlatma yok
      recognitionRef.current = null;
      setIsListening(false);
    };

    try {
      rec.start();
      recognitionRef.current = rec;
      setIsListening(true);
    } catch (e) {
      console.error("Recognition start error:", e);
    }
  };

  // --- Dinlemeyi durdur ---
  const stopListening = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {}
      recognitionRef.current = null;
    }
    setIsListening(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    stopListening();
    speechSynthesis.cancel();
    setMessages([{ role: "assistant", content: "Yeni bir sohbete başlayalım! ✨" }]);
    setInput("");
  };

  return (
    <div className="page">
      <div className="chat">
        <header className="chat__header">
          <div className="title">AthenAI</div>
          <div className="controls">
            {!isListening ? (
              <button className="btn btn--primary" onClick={startListening}>🎤 Başlat</button>
            ) : (
              <button className="btn btn--danger" onClick={stopListening}>⏹ Durdur</button>
            )}
            <button className="btn" onClick={clearChat}>🧹 Yeni sohbet</button>
          </div>
        </header>

        <div className="chat__messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role === "user" ? "msg--user" : "msg--bot"}`}>
              <div className="avatar">{m.role === "user" ? "🧑" : "🤖"}</div>
              <div className="bubble">{m.content}</div>
            </div>
          ))}

          {isThinking && (
            <div className="msg msg--bot">
              <div className="avatar">🤖</div>
              <div className="bubble bubble--thinking">
                <span className="dot"></span><span className="dot"></span><span className="dot"></span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="chat__input">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Mesaj yazın… (Göndermek için Enter)"
          />
          <button className="btn btn--primary" onClick={() => handleSend()}>Gönder ➤</button>
        </div>
      </div>
    </div>
  );
}
