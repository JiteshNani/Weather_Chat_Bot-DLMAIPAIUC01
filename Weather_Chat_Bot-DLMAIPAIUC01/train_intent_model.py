from __future__ import annotations

import os
import json
import pickle
import random
from typing import Dict, List, Tuple

import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
MIN_STOP = {"a","an","the","is","are","was","were","in","on","at","to","for","of","and","or","please","tell","me","what","whats","what's","give","show","can","could","will"}

TRAIN_PATH = os.path.join("data", "training_intents.json")
OUT_PATH = os.path.join("models", "intent_model.pkl")

def tokenize(text: str):
    import re
    return re.findall(r"[a-zA-Z']+", text.lower())

def preprocess(text: str):
    toks = [t for t in tokenize(text) if t not in MIN_STOP]
    return [stemmer.stem(t) for t in toks]

def featurize(tokens):
    return {t: True for t in tokens}

def load_dataset() -> List[Tuple[Dict[str,bool], str]]:
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = []
    for intent, examples in data.items():
        for ex in examples:
            dataset.append((featurize(preprocess(ex)), intent))
    random.shuffle(dataset)
    return dataset

def main():
    os.makedirs("models", exist_ok=True)

    dataset = load_dataset()
    if len(dataset) < 20:
        raise SystemExit("Not enough training samples. Add more examples to training_intents.json")

    # Simple train-test split
    split = int(0.8 * len(dataset))
    train_set = dataset[:split]
    test_set = dataset[split:]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(classifier, test_set) if test_set else 1.0

    with open(OUT_PATH, "wb") as f:
        pickle.dump(classifier, f)

    print(f"Saved model to {OUT_PATH}")
    print(f"Validation accuracy (simple split): {acc:.2f}")
    classifier.show_most_informative_features(10)

if __name__ == "__main__":
    main()
