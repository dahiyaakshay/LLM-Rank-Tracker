import os
import json
import re
import requests
import time
from typing import Dict, List, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Try to download nltk data, if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

def query_groq_llm(prompt: str, model: str = "llama3-8b-8192", temperature: float = 0.2) -> str:
    """
    Query the Groq LLM API with a prompt and return the response.
    
    Args:
        prompt: The text prompt to send to the LLM
        model: The model to use (defaults to llama3-8b-8192)
        temperature: Controls randomness (0.0 to 1.0)
        
    Returns:
        The text response from the LLM
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Please provide your API key.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Endpoint for Groq's API
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    
    # Try up to 3 times with exponential backoff
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            return f"Error querying Groq API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Groq API response: {str(e)}"


def extract_brand_mentions(text: str, target_brand: str) -> Dict[str, Any]:
    """
    Extract and analyze brand mentions from the LLM response using more sophisticated methods.
    
    Args:
        text: The text response from the LLM
        target_brand: The brand name we're primarily interested in
        
    Returns:
        Dictionary with analysis of brand mentions
    """
    # Case-insensitive search for the target brand
    target_brand_lower = target_brand.lower()
    text_lower = text.lower()
    
    # Check if the target brand is mentioned
    # Use word boundary check to avoid partial matches
    pattern = r'\b' + re.escape(target_brand_lower) + r'\b'
    is_mentioned = bool(re.search(pattern, text_lower))
    
    # Count mentions of the target brand
    mention_count = len(re.findall(pattern, text_lower))
    
    # Extract context around mentions (full sentences containing the brand)
    mention_contexts = []
    if is_mentioned:
        # Tokenize text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if re.search(pattern, sentence.lower()):
                mention_contexts.append(sentence.strip())
    
    # Determine rank position 
    rank = -1
    if is_mentioned:
        # Look for numbered lists, bullet points or ranking language
        # First split by newlines to handle list items
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if re.search(pattern, line_lower):
                # Check if line starts with a number or bullet
                if re.match(r'^\s*(\d+[\.\)]|\*|\-)\s+', line):
                    match = re.match(r'^\s*(\d+)[\.\)]', line)
                    if match:
                        rank = int(match.group(1))
                        break
                    else:
                        # Count previous bullet points to estimate rank
                        bullet_count = 0
                        for j in range(i):
                            if re.match(r'^\s*(\*|\-|\d+[\.\)])\s+', lines[j]):
                                bullet_count += 1
                        rank = bullet_count + 1
                        break
                
                # Look for ranking language in the line
                ranking_terms = ["#1", "#2", "#3", "#4", "#5", 
                                 "first", "second", "third", "fourth", "fifth",
                                 "1st", "2nd", "3rd", "4th", "5th",
                                 "top choice", "winner", "best option"]
                
                for idx, term in enumerate(ranking_terms):
                    if term in line_lower:
                        # Simple mapping based on ranking term
                        if idx < 5:  # #1 through #5
                            rank = idx + 1
                        elif idx < 10:  # first through fifth
                            rank = idx - 4
                        elif idx < 15:  # 1st through 5th
                            rank = idx - 9
                        else:  # general terms like "top choice"
                            rank = 1
                        break
        
        # If rank not found but brand is mentioned, provide a rough estimate
        if rank == -1:
            # Search for the brand in a context of ranking
            rank_context_patterns = [
                r'(top|best|leading|popular|recommended).*\b' + re.escape(target_brand_lower) + r'\b',
                r'\b' + re.escape(target_brand_lower) + r'\b.*is (top|best|leading|popular|recommended)'
            ]
            
            for pattern in rank_context_patterns:
                if re.search(pattern, text_lower):
                    rank = 3  # Approximate middle rank
                    break
            
            # If still not found, make an estimate based on position
            if rank == -1:
                # Simple heuristic: position of first mention divided by total length
                first_mention_pos = text_lower.find(target_brand_lower)
                if first_mention_pos > 0:
                    # Rough estimate based on position in text (1-10 scale)
                    rank = max(1, min(10, int(10 * (first_mention_pos / len(text_lower)))))
    
    # Extract other brands mentioned 
    other_brands = {}
    
    # Define patterns to find potential brand names
    brand_patterns = [
        # Capitalized words (possibly multi-word brands)
        r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b',
        # Quoted names that might be brands
        r'"([^"]+)"',
        # Bold text in markdown that might indicate brand names
        r'\*\*([^*]+)\*\*',
        # Names followed by typical brand indicators
        r'([A-Za-z0-9]+)(?:\s+Inc\.|\s+LLC|\s+Ltd\.|\s+Corporation|\s+Co\.)' 
    ]
    
    # For storing potential brand names before filtering
    potential_brands = {}
    
    # Extract potential brands using patterns
    for pattern in brand_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from capture group if it's a tuple
                
            # Skip single letters, short strings, or strings with special characters
            if (len(match) < 3 or 
                match.lower() == target_brand_lower or
                re.search(r'[,;:()\[\]{}]', match) or
                match.lower() in ['the', 'and', 'for', 'with', 'that', 'this', 'these', 'those']):
                continue
            
            # Count as a potential brand
            potential_brands[match] = text.count(match)
    
    # Filter for brands that appear multiple times or in a list context
    for brand, count in potential_brands.items():
        brand_lower = brand.lower()
        
        # Check if it appears in a list context (numbered or bulleted)
        in_list_context = False
        lines = text.split('\n')
        for line in lines:
            if brand in line and re.match(r'^\s*(\d+[\.\)]|\*|\-)\s+', line):
                in_list_context = True
                break
        
        # Include if it appears multiple times or in a list context
        if count > 1 or in_list_context:
            other_brands[brand] = count
    
    # Sort other brands by mention count and limit to top 15
    other_brands = dict(sorted(other_brands.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Perform sentiment analysis
    sentiment = "neutral"
    sentiment_score = 0.0
    
    if is_mentioned and mention_contexts:
        try:
            # Use NLTK's VADER for sentiment analysis
            sia = SentimentIntensityAnalyzer()
            
            # Analyze sentiment for each mention context
            sentiment_scores = []
            for context in mention_contexts:
                score = sia.polarity_scores(context)
                sentiment_scores.append(score['compound'])
            
            # Calculate average sentiment score
            if sentiment_scores:
                sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
                
                # Map to sentiment categories
                if sentiment_score >= 0.05:
                    sentiment = "positive"
                elif sentiment_score <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
        except Exception:
            # Fallback to simple keyword-based sentiment analysis
            positive_words = ["best", "excellent", "top", "leading", "recommended", "popular", 
                             "great", "innovative", "powerful", "favorite", "outstanding", "superior"]
            negative_words = ["worst", "poor", "avoid", "limited", "expensive", "overpriced",
                             "difficult", "problematic", "disappointing", "mediocre", "unreliable"]
            
            positive_count = 0
            negative_count = 0
            
            for context in mention_contexts:
                context_lower = context.lower()
                for word in positive_words:
                    if f" {word} " in f" {context_lower} ":
                        positive_count += 1
                for word in negative_words:
                    if f" {word} " in f" {context_lower} ":
                        negative_count += 1
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
    
    return {
        "is_mentioned": is_mentioned,
        "mention_count": mention_count,
        "rank": rank,
        "mention_contexts": mention_contexts,
        "other_brands": other_brands,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score
    }
