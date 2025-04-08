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


def extract_brand_mentions(text: str, target_brand: str, category: str = "") -> Dict[str, Any]:
    """
    Extract and analyze brand mentions from the LLM response using category-aware filtering.
    
    Args:
        text: The text response from the LLM
        target_brand: The brand name we're primarily interested in
        category: The category context (e.g., "smartphones", "cars", etc.)
        
    Returns:
        Dictionary with analysis of brand mentions
    """
    # Define category-specific known brands
    category_brands = {
        "smartphone": [
            "iphone", "samsung", "google", "pixel", "huawei", "xiaomi", "oneplus", "oppo", 
            "vivo", "motorola", "sony", "nokia", "lg", "asus", "honor", "realme", "htc", 
            "blackberry", "apple", "nothing", "poco", "tecno", "infinix", "zte", "lenovo"
        ],
        "laptop": [
            "macbook", "dell", "hp", "lenovo", "asus", "acer", "microsoft", "surface", 
            "samsung", "msi", "razer", "lg", "toshiba", "huawei", "xiaomi", "vaio", 
            "alienware", "chromebook", "thinkpad", "apple", "google"
        ],
        "car": [
            "toyota", "honda", "ford", "chevrolet", "tesla", "bmw", "mercedes", "audi", 
            "volkswagen", "hyundai", "kia", "nissan", "lexus", "mazda", "subaru", "porsche", 
            "ferrari", "lamborghini", "jaguar", "volvo", "land rover", "jeep", "dodge", 
            "chrysler", "cadillac", "buick", "acura", "infiniti", "maserati", "bentley", 
            "rolls royce", "aston martin", "mini", "fiat", "alfa romeo", "peugeot", "renault", 
            "citroen", "skoda", "seat"
        ],
        "tv": [
            "samsung", "lg", "sony", "vizio", "tcl", "hisense", "panasonic", "philips", 
            "sharp", "toshiba", "insignia", "element", "sceptre", "rca", "onn", "skyworth",
            "westinghouse", "jvc", "hitachi", "haier", "polaroid", "Thomson", "xiaomi"
        ],
        "streaming": [
            "netflix", "hulu", "disney", "hbo", "amazon", "prime", "peacock", "paramount", 
            "apple tv", "discovery", "youtube", "crunchyroll", "tubi", "roku", "pluto", 
            "fubo", "sling", "crackle", "starz", "showtime", "espn", "dazn", "max"
        ],
        "ecommerce": [
            "amazon", "ebay", "walmart", "etsy", "shopify", "alibaba", "aliexpress", 
            "wayfair", "target", "best buy", "home depot", "newegg", "overstock", "wish", 
            "temu", "shein", "zalando", "asos", "zappos", "rakuten"
        ],
    }
    
    # Common words and terms that shouldn't be identified as brands
    common_words = {
        'high', 'low', 'key', 'top', 'best', 'better', 'worse', 'good', 'great', 'excellent', 
        'poor', 'bad', 'average', 'medium', 'large', 'small', 'fast', 'slow', 'expensive', 'cheap',
        'budget', 'premium', 'quality', 'performance', 'value', 'price', 'cost', 'feature', 'features',
        'advantage', 'advantages', 'disadvantage', 'disadvantages', 'benefit', 'benefits', 'target',
        'user', 'users', 'consumer', 'consumers', 'customer', 'customers', 'market', 'marketing',
        'technology', 'product', 'products', 'service', 'services', 'device', 'devices',
        'option', 'options', 'alternative', 'alternatives', 'competitor', 'competitors',
        'design', 'functionality', 'durability', 'reliability', 'efficiency', 'innovation', 
        'sleek', 'modern', 'traditional', 'limited', 'advanced', 'basic', 'standard', 'premium',
        'overall', 'generally', 'specifically', 'notably', 'additionally', 'similarly',
        'however', 'therefore', 'moreover', 'furthermore', 'consequently', 'regardless',
        'overview', 'summary', 'conclusion', 'introduction', 'analysis', 'assessment',
        'report', 'guide', 'review', 'comparison', 'evaluation', 'recommendation',
        'interface', 'experience', 'usability', 'accessibility', 'compatibility',
        'popularity', 'availability', 'affordability', 'versatility', 'flexibility',
        'the', 'and', 'for', 'with', 'that', 'this', 'these', 'those', 'they', 'them', 'their',
        'what', 'when', 'where', 'why', 'how', 'who', 'which', 'would', 'could', 'should', 'will',
        'can', 'may', 'might', 'must', 'shall', 'are', 'have', 'has', 'had', 'was', 'were', 'been',
        'thing', 'things', 'time', 'times', 'place', 'places', 'person', 'people', 'way', 'ways',
        'day', 'days', 'month', 'months', 'year', 'years', 'world', 'country', 'countries',
        'city', 'cities', 'town', 'towns', 'area', 'areas', 'region', 'regions', 'location',
        'model', 'models', 'version', 'versions', 'edition', 'editions', 'series', 'level',
        'levels', 'tier', 'tiers', 'generation', 'generations', 'class', 'classes', 'grade',
        'grades', 'rating', 'ratings', 'rank', 'ranks', 'ranking', 'rankings', 'score', 'scores',
        'powered', 'ai-powered', 'ai', 'ml', 'tech', 'smart', 'intelligent', 'eco', 'eco-friendly',
        'green', 'sustainable', 'environment', 'environmental', 'digital', 'analog', 'manual',
        'automatic', 'automated', 'connected', 'wireless', 'wired', 'portable', 'compact',
        'lightweight', 'heavy', 'durable', 'fragile', 'robust', 'delicate', 'sensitive', 'responsive',
        'unresponsive', 'intuitive', 'complicated', 'simple', 'complex', 'easy', 'difficult',
        'hard', 'soft', 'active', 'passive', 'dynamic', 'static', 'flexible', 'rigid', 'agile',
        'sturdy', 'flimsy', 'thin', 'thick', 'wide', 'narrow', 'deep', 'shallow', 'tall', 'short',
        'long', 'brief', 'quick', 'slow', 'rapid', 'gradual', 'instant', 'delayed', 'immediate',
        'eventual', 'current', 'previous', 'next', 'future', 'past', 'present', 'old', 'new',
        'ancient', 'modern', 'contemporary', 'traditional', 'conventional', 'unconventional',
        'typical', 'atypical', 'normal', 'abnormal', 'regular', 'irregular', 'common', 'uncommon',
        'rare', 'frequent', 'occasional', 'sporadic', 'constant', 'variable', 'fixed', 'adjustable',
        'customize', 'customizable', 'personalize', 'personalizable', 'generic', 'specific',
        'specialized', 'general', 'particular', 'unique', 'standard', 'nonstandard', 'universal',
        'global', 'local', 'regional', 'national', 'international', 'worldwide', 'domestic',
        'foreign', 'exported', 'imported', 'outsourced', 'insourced', 'proprietary', 'open',
        'closed', 'public', 'private', 'secure', 'insecure', 'safe', 'unsafe', 'dangerous',
        'harmless', 'toxic', 'nontoxic', 'healthy', 'unhealthy', 'organic', 'inorganic',
        'natural', 'artificial', 'synthetic', 'genuine', 'fake', 'authentic', 'counterfeit',
        'imitation', 'original', 'copy', 'clone', 'replica', 'prototype', 'concept', 'final',
        'complete', 'incomplete', 'partial', 'full', 'empty', 'half', 'quarter', 'third',
        'fraction', 'percentage', 'amount', 'segment', 'piece', 'part', 'component', 'element',
        'ingredient', 'constituent', 'substance', 'matter', 'material', 'fabric', 'texture',
        'composition', 'structure', 'construction', 'building', 'architecture', 'design',
        'blueprint', 'plan', 'scheme', 'project', 'program', 'system', 'setup', 'configuration',
        'arrangement', 'organization', 'management', 'administration', 'control', 'supervise',
        'regulate', 'rule', 'govern', 'coordinate', 'align', 'direct', 'lead', 'guide', 'steer',
        'pilot', 'navigate', 'belgian', 'european', 'american', 'asian', 'african', 'australian'
    }
    
    # Location-based terms that should not be treated as brands
    location_terms = {
        'europe', 'asia', 'north america', 'south america', 'africa', 'australia', 'antarctica', 
        'united states', 'usa', 'canada', 'mexico', 'brazil', 'argentina', 'colombia', 'peru', 'chile',
        'united kingdom', 'uk', 'england', 'scotland', 'wales', 'northern ireland', 'ireland', 
        'france', 'germany', 'italy', 'spain', 'portugal', 'netherlands', 'belgium', 'luxembourg',
        'switzerland', 'austria', 'poland', 'czech republic', 'slovakia', 'hungary', 'romania',
        'bulgaria', 'greece', 'turkey', 'russia', 'ukraine', 'belarus', 'estonia', 'latvia', 
        'lithuania', 'finland', 'sweden', 'norway', 'denmark', 'iceland', 'greenland',
        'china', 'japan', 'south korea', 'north korea', 'india', 'pakistan', 'bangladesh', 'nepal',
        'bhutan', 'sri lanka', 'maldives', 'thailand', 'myanmar', 'laos', 'cambodia', 'vietnam',
        'malaysia', 'singapore', 'indonesia', 'philippines', 'australia', 'new zealand', 'fiji',
        'papua new guinea', 'saudi arabia', 'uae', 'qatar', 'kuwait', 'bahrain', 'oman', 'yemen',
        'iran', 'iraq', 'syria', 'lebanon', 'israel', 'palestine', 'jordan', 'egypt', 'libya',
        'algeria', 'morocco', 'tunisia', 'sudan', 'south sudan', 'ethiopia', 'somalia', 'kenya',
        'uganda', 'tanzania', 'rwanda', 'burundi', 'congo', 'south africa', 'namibia', 'botswana',
        'zimbabwe', 'zambia', 'angola', 'nigeria', 'ghana', 'ivory coast', 'senegal', 'mali',
        'european', 'asian', 'american', 'african', 'australian', 'middle eastern', 'latin american',
        'north american', 'south american', 'british', 'french', 'german', 'italian', 'spanish',
        'portuguese', 'dutch', 'belgian', 'swiss', 'austrian', 'polish', 'czech', 'slovak', 'hungarian',
        'romanian', 'bulgarian', 'greek', 'turkish', 'russian', 'ukrainian', 'estonian', 'latvian',
        'lithuanian', 'finnish', 'swedish', 'norwegian', 'danish', 'icelandic', 'chinese', 'japanese',
        'korean', 'indian', 'pakistani', 'bangladeshi', 'nepali', 'sri lankan', 'thai', 'vietnamese',
        'malaysian', 'singaporean', 'indonesian', 'filipino', 'australian', 'new zealander', 'saudi',
        'emirati', 'qatari', 'kuwaiti', 'bahraini', 'omani', 'yemeni', 'iranian', 'iraqi', 'syrian',
        'lebanese', 'israeli', 'palestinian', 'jordanian', 'egyptian', 'libyan', 'algerian', 'moroccan',
        'tunisian', 'sudanese', 'ethiopian', 'somali', 'kenyan', 'ugandan', 'tanzanian', 'rwandan',
        'burundian', 'congolese', 'south african', 'namibian', 'botswanan', 'zimbabwean', 'zambian',
        'angolan', 'nigerian', 'ghanaian', 'ivorian', 'senegalese', 'malian'
    }
    
    # Combine common words and location terms
    filtered_terms = common_words.union(location_terms)
    
    # Determine the relevant category based on the target brand or provided category
    relevant_category = None
    target_brand_lower = target_brand.lower()
    
    # Try to identify category from the given category parameter
    if category:
        category_lower = category.lower()
        for cat in category_brands:
            if cat in category_lower or any(keyword in category_lower for keyword in ['phone', 'mobile', 'smartphone', 'cellular']):
                relevant_category = 'smartphone'
                break
            elif any(keyword in category_lower for keyword in ['laptop', 'computer', 'pc', 'notebook', 'desktop']):
                relevant_category = 'laptop'
                break
            elif any(keyword in category_lower for keyword in ['car', 'vehicle', 'automobile', 'suv', 'sedan', 'truck']):
                relevant_category = 'car'
                break
            elif any(keyword in category_lower for keyword in ['tv', 'television', 'display', 'screen', 'monitor']):
                relevant_category = 'tv'
                break
            elif any(keyword in category_lower for keyword in ['streaming', 'video', 'platform', 'entertainment', 'media']):
                relevant_category = 'streaming'
                break
            elif any(keyword in category_lower for keyword in ['ecommerce', 'shopping', 'retail', 'online store', 'marketplace']):
                relevant_category = 'ecommerce'
                break
    
    # If category not found, try to determine from target brand
    if not relevant_category:
        for cat, brands in category_brands.items():
            if target_brand_lower in brands:
                relevant_category = cat
                break
    
    # Default to smartphone if no category could be determined
    if not relevant_category:
        relevant_category = 'smartphone'
    
    # Get the list of known brands for the identified category
    known_brands = category_brands.get(relevant_category, [])
    
    # Case-insensitive search for the target brand
    # Use word boundary check to avoid partial matches
    pattern = r'\b' + re.escape(target_brand_lower) + r'\b'
    is_mentioned = bool(re.search(pattern, text.lower()))
    
    # Count mentions of the target brand
    mention_count = len(re.findall(pattern, text.lower()))
    
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
                if re.search(pattern, text.lower()):
                    rank = 3  # Approximate middle rank
                    break
            
            # If still not found, make an estimate based on position
            if rank == -1:
                # Simple heuristic: position of first mention divided by total length
                first_mention_pos = text.lower().find(target_brand_lower)
                if first_mention_pos > 0:
                    # Rough estimate based on position in text (1-10 scale)
                    rank = max(1, min(10, int(10 * (first_mention_pos / len(text.lower())))))
    
    # Extract other brands mentioned - focused on the relevant category
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
                match.lower() in filtered_terms):
                continue
            
            # Count as a potential brand
            potential_brands[match] = text.count(match)
    
    # Filter for brands that appear multiple times or in a list context
    # Prioritize known brands in the same category
    for brand, count in list(potential_brands.items()):
        brand_lower = brand.lower()
        
        # Skip common words and location terms
        if brand_lower in filtered_terms:
            del potential_brands[brand]
            continue
        
        # Skip if it's just a single word that's common
        if len(brand.split()) == 1 and brand_lower in filtered_terms:
            del potential_brands[brand]
            continue
        
        # Check if it appears in a list context (numbered or bulleted)
        in_list_context = False
        lines = text.split('\n')
        for line in lines:
            if brand in line and re.match(r'^\s*(\d+[\.\)]|\*|\-)\s+', line):
                in_list_context = True
                break
        
        # Boost score for known brands in the same category
        is_known_brand = brand_lower in known_brands or any(brand_lower in kb for kb in known_brands)
        
        # Adjust the threshold based on whether it's a known brand
        min_mentions = 1 if is_known_brand else 2
        
        # Include if it's a known brand, appears multiple times, or in a list context
        if not (is_known_brand or count >= min_mentions or in_list_context):
            del potential_brands[brand]
    
    # Sort the remaining brands, first by known status, then by mention count
    def brand_sort_key(item):
        brand, count = item
        brand_lower = brand.lower()
        is_known = brand_lower in known_brands or any(brand_lower in kb for kb in known_brands)
        return (-1 if is_known else 0, count)
    
    # Convert the filtered brands to the output format
    sorted_brands = sorted(potential_brands.items(), key=brand_sort_key, reverse=True)
    other_brands = {brand: count for brand, count in sorted_brands[:15]}
    
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
        "sentiment_score": sentiment_score,
        "category": relevant_category
    }
