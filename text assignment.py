import nltk
import subprocess
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import spacy
from collections import Counter

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Ensure the spaCy model is available
def ensure_spacy_model():
    try:
        # Try to load the spaCy model
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, download it
        print("Downloading spaCy model 'en_core_web_sm'...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("Download complete. Model 'en_core_web_sm' is now available.")

ensure_spacy_model()

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# File paths
files = ["Text_1.txt", "Text_2.txt", "Text_3.txt", "Text_4.txt"]

# Function to read text from a file
def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Tokenization, Stemming, Lemmatization, and NER Analysis
def analyze_text(file_path):
    text = read_text(file_path)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Frequency Distribution
    freq_dist = FreqDist(tokens)
    most_common_tokens = freq_dist.most_common(20)
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Named Entity Recognition
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    
    return {
        "most_common_tokens": most_common_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "named_entity_count": len(named_entities),
        "named_entities": named_entities
    }

# N-gram Analysis
def ngram_analysis(file_path, n=3):
    text = read_text(file_path)
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    ngram_freq = Counter(n_grams)
    most_common_ngrams = ngram_freq.most_common(10)
    return most_common_ngrams

# Perform analysis on all texts
results = {}
for file in files:
    if file != "Text_4.txt":
        results[file] = analyze_text(file)
    else:
        results[file] = ngram_analysis(file)

# Display results
for file, result in results.items():
    if file != "Text_4.txt":
        print(f"\nResults for {file}:")
        print("Most Common Tokens:", [token for token, _ in result["most_common_tokens"]])
        print("Named Entity Count:", result["named_entity_count"])
    else:
        print(f"\nResults for {file}:")
        print("Most Common N-grams:", [" ".join(ngram) for ngram, _ in result])