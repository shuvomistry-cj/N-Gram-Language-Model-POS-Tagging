import nltk
import random
import math
import streamlit as st
from nltk import word_tokenize, ngrams
from collections import defaultdict
from nltk.corpus import brown
from nltk.tag import DefaultTagger, UnigramTagger
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_treebank_pos_tagger')  # Ensure this is downloaded
nltk.download('tagsets')  # Optional, helps with tag descriptions


# ---------------------- Streamlit Frontend ---------------------- #
st.title("N-Gram Language Model & POS Tagging")

# User input for corpus
corpus = st.text_area("Enter Text Corpus:", "This is a simple corpus. This corpus is small but useful.")
tokens = word_tokenize(corpus.lower())

# User selection for N-gram size and smoothing
n = st.selectbox("Select N-gram size:", [2, 3])
smoothing = st.checkbox("Enable Smoothing")

def build_ngram_model(tokens, n=2, smoothing=False):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    n_grams = list(ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
    for gram in n_grams:
        prefix, word = tuple(gram[:-1]), gram[-1]
        model[prefix][word] += 1
    
    for prefix in model:
        total_count = float(sum(model[prefix].values()))
        vocab_size = len(set(tokens)) + 1 if smoothing else 0
        for word in model[prefix]:
            model[prefix][word] = (model[prefix][word] + 1) / (total_count + vocab_size) if smoothing else model[prefix][word] / total_count
    return model

# Building and displaying N-gram model
ngram_model = build_ngram_model(tokens, n, smoothing)
st.write("N-gram Model:", dict(ngram_model))

def perplexity(model, test_tokens, n=2):
    test_grams = list(ngrams(test_tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
    log_prob_sum = 0
    N = len(test_grams)
    for gram in test_grams:
        prefix, word = tuple(gram[:-1]), gram[-1]
        prob = model[prefix].get(word, 1e-6)  # Avoid zero probability
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / N)

# Compute and display perplexity
if st.button("Calculate Perplexity"):
    test_tokens = word_tokenize(corpus.lower())
    perplexity_score = perplexity(ngram_model, test_tokens, n)
    st.write("Perplexity:", perplexity_score)

# ---------------------- Part-of-Speech Tagging ---------------------- #
st.header("Part-of-Speech Tagging")

# Ensure the required NLTK resource is available
nltk.download('averaged_perceptron_tagger')

test_sentence = st.text_input("Enter a sentence for POS tagging:", "This is a simple corpus used for tagging")
test_tokens = word_tokenize(test_sentence)  # Keep original case

# Rule-Based POS Tagger
def simple_rule_based_tagger(words):
    rules = {"is": "VERB", "am": "VERB", "are": "VERB", "was": "VERB", "were": "VERB", "the": "DET", "a": "DET", "an": "DET", "corpus": "NOUN", "model": "NOUN"}
    return [(word, rules.get(word.lower(), "NOUN")) for word in words]

# Pre-trained Stochastic Tagger
brown_tagged_sents = brown.tagged_sents(categories='news', tagset='universal')
tagger = UnigramTagger(brown_tagged_sents, backoff=DefaultTagger('NOUN'))

# Tagging outputs
rule_based_tags = simple_rule_based_tagger(test_tokens)
stochastic_tags = tagger.tag(test_tokens)

try:
    nltk_tags = pos_tag(test_tokens, tagset='universal')
except LookupError:
    nltk_tags = "Error: NLTK tagger resource not found. Please check your nltk.download calls."

st.write("Rule-Based POS Tags:", rule_based_tags)
st.write("Stochastic POS Tags:", stochastic_tags)
st.write("NLTK POS Tags:", nltk_tags)

