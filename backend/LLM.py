from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
import json
import re
from backend.PromptTemplate import get_recommendation_prompt

class LLMHandler:
    def __init__(self):
        load_dotenv()
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )

    def get_recommendation(self, user_query, movie_infos):
        prompt = get_recommendation_prompt(user_query, movie_infos)
        response = self.llm.invoke(prompt)
        print('LLM response:', response)
        # Remove code block markers and extract JSON array
        resp = response.strip()
        # Remove triple backticks and optional 'json' at start/end
        resp = re.sub(r'^```(?:json)?', '', resp, flags=re.IGNORECASE).strip()
        resp = re.sub(r'```$', '', resp).strip()
        # Extract the first JSON array from the response
        match = re.search(r'(\[.*?\])', resp, re.DOTALL)
        if match:
            resp = match.group(1)
        try:
            recommendations = json.loads(resp)
        except Exception as e:
            print("Failed to parse LLM response as JSON:", e)
            recommendations = []
        return recommendations 