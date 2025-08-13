import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Bünyamin\Desktop\chatbot-translate-468818-59d06963ad59.json"

from google.cloud import translate

class Translator:
    def __init__(self, project_id, location="global"):
        self.client = translate.TranslationServiceClient()
        self.project_id = project_id
        self.location = location
        self.parent = f"projects/{self.project_id}/locations/{self.location}"

    def detect_language(self, text):
        response = self.client.detect_language(
            request={
                "parent": self.parent,
                "content": text,
                "mime_type": "text/plain"
            }
        )
        return response.languages[0].language_code

    def translate_text(self, text, target_language):
        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "target_language_code": target_language,
            }
        )
        return response.translations[0].translated_text
