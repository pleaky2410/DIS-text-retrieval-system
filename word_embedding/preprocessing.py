import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

def preprocess_text(text):
	# Convert text to lowercase
	text = text.lower()
	# Remove special characters and digits using regular expressions
	text = re.sub(r'\d+', '', text)
	# Remove digits
	text = re.sub(r'[^\w\s]', '', text) # Remove special characters
	# Tokenize the text
	tokens = nltk.word_tokenize(text)
	return tokens

def remove_stopwords (tokens) :
	stop_words = set(stopwords.words('english'))
	filtered_tokens = [word for word in tokens if word not in stop_words]
	return filtered_tokens

def perform_lemmatization (tokens):
	lemmatizer = nltk.WordNetLemmatizer()
	lemmatized_tokens = [lemmatizer. lemmatize (token) for token in tokens]
	return lemmatized_tokens

def clean_text(text):
	tokens = preprocess_text(text)
	filtered_tokens = remove_stopwords(tokens)
	lemmatized_tokens = perform_lemmatization(filtered_tokens)
	clean_text = ' '.join(lemmatized_tokens)
	return clean_text

# text = "어떤 방식으로 췌몽상과 비상천에서 빗자루를 타고 돌진하는 형태로 사용되는 것인가요?"
# cleaned_text = clean_text(text)
# print(cleaned_text)