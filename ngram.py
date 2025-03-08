import nltk
import random
import math
from nltk import word_tokenize, bigrams, trigrams
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')

# Sample corpus for N-gram model
corpus = """This is a simple corpus. This corpus is small but useful.
            The purpose of this corpus is to build an N-gram model."""
tokens = word_tokenize(corpus.lower())

# ---------------------- N-GRAM LANGUAGE MODEL ---------------------- #

def build_ngram_model(tokens, n=2, smoothing=False):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    
    n_grams = list(ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
    
    for gram in n_grams:
        prefix, word = tuple(gram[:-1]), gram[-1]
        model[prefix][word] += 1
    
    # Convert counts to probabilities
    for prefix in model:
        total_count = float(sum(model[prefix].values()))
        if smoothing:
            vocab_size = len(set(tokens)) + 1  # Add-one smoothing
            for word in model[prefix]:
                model[prefix][word] = (model[prefix][word] + 1) / (total_count + vocab_size)
        else:
            for word in model[prefix]:
                model[prefix][word] /= total_count
    
    return model

# Building unsmoothed and smoothed models
bigram_model_unsmoothed = build_ngram_model(tokens, n=2, smoothing=False)
bigram_model_smoothed = build_ngram_model(tokens, n=2, smoothing=True)

# Perplexity Calculation
def perplexity(model, test_tokens, n=2):
    test_grams = list(ngrams(test_tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
    
    log_prob_sum = 0
    N = len(test_grams)
    
    for gram in test_grams:
        prefix, word = tuple(gram[:-1]), gram[-1]
        prob = model[prefix].get(word, 1e-6)  # Avoid zero probability
        log_prob_sum += math.log(prob)
    
    return math.exp(-log_prob_sum / N)

# Testing with the same corpus
test_tokens = word_tokenize(corpus.lower())
print("Perplexity (Unsmoothed Bigram Model):", perplexity(bigram_model_unsmoothed, test_tokens, n=2))
print("Perplexity (Smoothed Bigram Model):", perplexity(bigram_model_smoothed, test_tokens, n=2))

# ---------------------- PART-OF-SPEECH TAGGING ---------------------- #

# Simple Rule-Based POS Tagger
def simple_rule_based_tagger(words):
    rules = {"is": "VERB", "am": "VERB", "are": "VERB", "was": "VERB", "were": "VERB", 
             "the": "DET", "a": "DET", "an": "DET", "corpus": "NOUN", "model": "NOUN"}
    return [(word, rules.get(word.lower(), "NOUN")) for word in words]

# Pre-trained Stochastic Tagger
brown_tagged_sents = brown.tagged_sents(categories='news', tagset='universal')
tagger = UnigramTagger(brown_tagged_sents, backoff=DefaultTagger('NOUN'))

test_sentence = "This is a simple corpus used for tagging"
test_tokens = word_tokenize(test_sentence.lower())

# Comparing taggers
rule_based_tags = simple_rule_based_tagger(test_tokens)
stochastic_tags = tagger.tag(test_tokens)
nltk_tags = pos_tag(test_tokens, tagset='universal')

print("\nRule-Based POS Tags:", rule_based_tags)
print("\nStochastic POS Tags:", stochastic_tags)
print("\nNLTK POS Tags:", nltk_tags)
