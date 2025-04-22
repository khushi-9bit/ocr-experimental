import spacy
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import hdbscan
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def generate_summary(conversation):
    # Combine all messages into a single text block
    full_text = " ".join(conversation)

    # Use Sumy's LexRank summarizer
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    # Generate a summary with 3 sentences
    summary_sentences = summarizer(parser.document, 3)

    # Convert summary sentences to a string
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    return summary if summary else "Summary generation failed. The conversation might be too short."

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to preprocess and clean the conversation
def preprocess_conversation(conversation):
    return [text.lower() for text in conversation]  # Lowercasing for uniform processing

# Function for Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]

# Function for Topic Modeling using LDA
def extract_topics(texts, num_topics=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)
    
    topics = []
    for index, topic in enumerate(lda_model.components_):
        topic_keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics.append(" ".join(topic_keywords))
    
    return topics

# Function for Topic Modeling using BERTopic
def extract_topics_bertopic(texts):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)  # Ensure valid parameters
    topic_model = BERTopic(hdbscan_model=hdbscan_model)
    
    topics, _ = topic_model.fit_transform(texts)
    return topic_model.get_topic_info()

# Function to extract action words (verbs)
def extract_actions(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ == "VERB"]

# Function to analyze conversation while keeping outputs intact
def analyze_conversation(conversation):
    cleaned_conversations = preprocess_conversation(conversation)

    # Extract topics using LDA
    lda_topics = extract_topics(cleaned_conversations)

    # Extract topics using BERTopic
    bertopic_topics = extract_topics_bertopic(cleaned_conversations)

    # Extract named entities (NER) and actions
    all_entities = []
    all_actions = []

    for text in cleaned_conversations:
        entities = extract_entities(text)
        actions = extract_actions(text)

        all_entities.extend(entities)
        all_actions.extend(actions)

        print("\n--- Entities (NER) ---")
        print(entities)

        print("\n--- Actions ---")
        print(actions)

    # Print the summary **separately at the end**
    print("\n--- Conversation Summary ---")
    print(generate_summary(conversation))

# Example Conversation (List of multiple texts)
conversation = [
    "Hey, how are you today? I'm doing great, just working on some new projects. Have you seen the latest trends in AI?",
    "I’ve been really interested in generative AI models lately. They are revolutionizing content creation! Do you think they will replace writers?",
    "I think AI can assist writers, but it won’t replace them. Humans bring creativity and emotions to their work. Have you used GPT models for writing tasks?",
    "Yes, I’ve been using GPT-3 for generating ideas and drafting content. It’s amazing how it can generate coherent paragraphs based on simple prompts.",
    "I’ve been reading a lot about legal AI applications. How do you think AI will impact the legal industry? It’s definitely a growing field.",
    "AI has a huge potential in legal research and document review. It can speed up workflows and increase accuracy. But there are concerns about data privacy.",
    "What about ethical concerns? With AI getting smarter, should we worry about its decision-making abilities in critical areas like healthcare or law?",
    "That’s a valid point. We need to make sure AI is developed responsibly, with transparency and accountability. Regulations will be key to preventing misuse."
]

# Analyze the conversation
analyze_conversation(conversation)
