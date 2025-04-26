import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
from coordinator import CoordinatorAgent
from dataset import dataset
from openai import OpenAI
from dotenv import load_dotenv
from api import api_key
from transformers import pipeline
import torch
import torch_directml


nltk.data.path.append(" C:\\Users\\saeed\\AppData\\Roaming\\nltk_data")

"""def text_generator(prompt, max_length=1000):
    model_id = "Qwen/Qwen2-0.5B-Instruct"

    # For Intel Iris Xe on Windows, DirectML is the best option
    try:
        dml = torch_directml.device()

        # Create pipeline with DirectML device and try passing truncation
        generator = pipeline('text-generation', model=model_id, device=dml, truncation=True, max_length=max_length)
    except ImportError:
        # Fallback to CPU if DirectML is not installed
        generator = pipeline('text-generation', model=model_id, device=-1, truncation=True, max_length=max_length)

    # Generate text
    response = generator(prompt, num_return_sequences=1)[0]['generated_text']
    return response"""

def fast_text_generator(prompt, max_length=1000):
    model_id = "Qwen/Qwen2-0.5B-Instruct"

    try:
        dml = torch_directml.device()
        generator = pipeline('text-generation', model=model_id, device=dml, truncation=True, max_length=max_length)
    except ImportError:
        generator = pipeline('text-generation', model=model_id, device=-1, truncation=True, max_length=max_length)

    # Prioritizing speed:
    generation_kwargs = {
        "num_return_sequences": 1,
        "max_length": max_length,
        "num_beams": 1,  # No beam search (greedy decoding) for speed
        "do_sample": False, # No sampling for deterministic and faster output
    }

    response = generator(prompt, **generation_kwargs)[0]['generated_text']
    return response


def tutor_agent(initial_prompt, max_iterations=2):
    # Start with the initial prompt
    current_response = fast_text_generator(initial_prompt)
    
    # Iterate to improve the response
    for i in range(max_iterations - 1):  # -1 because we already did the first iteration
        improvement_prompt = f"""
        I'll provide you with a prompt and your previous response to it.
        Please improve your response by making it more accurate, clear, comprehensive, and well-written.
        
        Original prompt: {initial_prompt}
        
        Your previous response: {current_response}
        
        Provide an improved version of your response:
        """
        
        new_response = fast_text_generator(improvement_prompt)
        
        # Check if we're getting back the improvement prompt and extract just the response
        if improvement_prompt in new_response:
            new_response = new_response.replace(improvement_prompt, "").strip()
        
        current_response = new_response
    
    return current_response

def summarizer_agent(prompt):
    # Check if any summarization-related words appear in the first ten words
    first_ten_words = prompt.split()[:10]
    first_ten_text = " ".join(first_ten_words).lower()
    
    summary_keywords = [
        "summarize", 
        "summary", 
        "summarization",
        "summarized",
        "summarizing"
    ]
    
    # Check if any of the summary keywords are in the first ten words
    should_skip_first_ten = any(keyword in first_ten_text for keyword in summary_keywords)
    
    # Remove the first ten words if they contain summary keywords
    if should_skip_first_ten and len(prompt.split()) > 10:
        cleaned_prompt = " ".join(prompt.split()[10:])
    else:
        cleaned_prompt = prompt
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Check if prompt is empty or too short
    if not cleaned_prompt or len(cleaned_prompt) < 20:
        return tutor_agent(prompt)    
    # Clean and tokenize text into sentences
    sentences = sent_tokenize(cleaned_prompt)
    
    # If there are very few sentences, return the original text
    if len(sentences) <= 2:
        return cleaned_prompt
    
    # Tokenize and preprocess words (stopword removal + lemmatization)
    def cnt_in_sent(sentences):
        text_data = []
        for i, sent in enumerate(sentences):
            words = word_tokenize(sent.lower())
            for word in words:
                if word.isalnum() and word not in stop_words:
                    word = lemmatizer.lemmatize(word)
                    text_data.append({'id': i + 1, 'word': word})
        return text_data
    
    # Frequency dictionary
    def freq1_dict(sentences):
        freq = {}
        for sent in sentences:
            words = word_tokenize(sent.lower())
            for word in words:
                if word.isalnum() and word not in stop_words:
                    word = lemmatizer.lemmatize(word)
                    freq[word] = freq.get(word, 0) + 1
        return freq
    
    # Calculate Term Frequency
    def calc_TF(text_data, freq_list):
        tf_scores = []
        for data in text_data:
            word = data['word']
            tf = freq_list[word]
            tf_scores.append({'id': data['id'], 'word': word, 'tf_score': tf})
        return tf_scores
    
    # Inverse Document Frequency
    def calc_IDF(text_data, freq_list):
        N = len(set([item['id'] for item in text_data]))  # number of sentences
        word_sent_map = {}
        for item in text_data:
            word_sent_map.setdefault(item['word'], set()).add(item['id'])
    
        idf_scores = []
        for word in freq_list:
            df = len(word_sent_map[word])
            idf = math.log((N + 1) / (df + 1)) + 1
            idf_scores.append({'word': word, 'idf_score': idf})
        return idf_scores
    
    # TF-IDF Scores
    def calc_TFIDF(tf_scores, idf_scores):
        idf_map = {item['word']: item['idf_score'] for item in idf_scores}
        tfidf = []
        for item in tf_scores:
            tfidf_score = item['tf_score'] * idf_map[item['word']]
            tfidf.append({'id': item['id'], 'word': item['word'], 'tfidf_score': tfidf_score})
        return tfidf
    
    # Sentence scoring with early-sentence boost
    def sent_scores(tfidf_scores, sentences, text_data, normalize=True):
        sent_data = []
        for i in range(1, len(sentences) + 1):  # Loop through each sentence ID
            sentence = sentences[i - 1]
            score = 0
            
            # Sum TF-IDF scores for all words in this sentence
            for t_dict in tfidf_scores:
                if t_dict['id'] == i:
                    score += t_dict['tfidf_score']
            
            if normalize:
                word_count = len(word_tokenize(sentence))
                score = score / word_count if word_count > 0 else score
            
            # Add position bias (earlier sentences get higher boost)
            position_boost = 0.1 / i  # Higher boost for earlier sentences
            score += position_boost
            
            sent_data.append({
                'id': i,
                'score': score,
                'sentence': sentence
            })
        
        return sent_data
    
    # Summary generator with sentence limit
    def summary(sent_data, sentences, max_percentage=0.5):
        sorted_data = sorted(sent_data, key=lambda x: x['score'], reverse=True)
        
        num_sentences = int(len(sentences) * max_percentage)
        num_sentences = min(num_sentences, len(sorted_data))
        num_sentences = max(num_sentences, 1)  # Ensure at least one sentence
        
        selected = sorted_data[:num_sentences]  # Select top scoring sentences
        
        # Sort by original position in the text
        selected = sorted(selected, key=lambda x: x['id'])
        
        formatted_summary = "\n\n".join(d['sentence'] for d in selected)
        
        return formatted_summary
    
    # Apply the summarization pipeline
    text_data = cnt_in_sent(sentences)
    freq_list = freq1_dict(sentences)
    
    # Check if we have enough content to summarize
    if not text_data or not freq_list:
        return "Could not extract meaningful content for summarization."
    
    tf_scores = calc_TF(text_data, freq_list)
    idf_scores = calc_IDF(text_data, freq_list)
    tfidf_scores = calc_TFIDF(tf_scores, idf_scores)
    sent_data = sent_scores(tfidf_scores, sentences, text_data, normalize=True)
    
    # Determine summary length based on input length
    if len(sentences) > 50:
        max_percentage = 0.3  # More aggressive summarization for longer texts
    else:
        max_percentage = 0.5  # Less aggressive for shorter texts
    
    result = summary(sent_data, sentences, max_percentage)
    
    return result

class MultiAgentSystem:
    def __init__(self):
        self.coordinator = None
        self.model_path = "C:\\Users\\saeed\\OneDrive\\Desktop\\Uni-files\\CSC-361\\code\\coordinator_model.keras"
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _ensure_model_loaded(self):
        if self.coordinator is not None:
            return
        
        self.coordinator = CoordinatorAgent()

        if os.path.exists(self.model_path):
            try:
                self.coordinator.load(self.model_path)
            except Exception:
                self._train_default_model()
        else:
            self._train_default_model()

    def _train_default_model(self):
        from dataset import dataset  # Assuming dataset is defined in dataset.py
        self.coordinator.train(dataset)
        self.coordinator.save(self.model_path)

    def handle_user_prompt(self, prompt):
        
        try:
            first_ten_words = prompt.split()[:10]
            first_ten_text = " ".join(first_ten_words).lower()
            
            summary_keywords = [
                "summarize", 
                "summary", 
                "summarization",
                "summarized",
                "summarizing"
            ]
            
            # Check if any of the summary keywords are in the first ten words
            should_skip_first_ten = any(keyword in first_ten_text for keyword in summary_keywords)
            
            # Remove the first ten words if they contain summary keywords
            if should_skip_first_ten and len(prompt.split()) > 10:
                cleaned_prompt = " ".join(prompt.split()[10:])
            else:
                cleaned_prompt = prompt
            
            print(cleaned_prompt)
            
            self._ensure_model_loaded()
            agent = self.coordinator.predict_agent(cleaned_prompt)
            
            if agent == "tutor":
                return tutor_agent(cleaned_prompt)
            elif agent == "summarizer":
                return summarizer_agent(cleaned_prompt)
            else:
                return "Unknown agent type."
        except Exception as e:
            return f"Error processing your request: {str(e)}"

if __name__ == "__main__":
    system = MultiAgentSystem()