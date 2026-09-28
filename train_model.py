"""Train the three classifiers (sentiment, emotion, category).

Model: TF-IDF (1-2 grams)  +  Logistic Regression, one pipeline per task.

Run:  python train_model.py

If data/feedback_dataset.csv does not exist, a labelled demo dataset is
generated from templates so the project works out of the box. To use real
data, replace that CSV with columns: text, sentiment, emotion, category.
"""
import os
import random

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nlp_utils import clean_text

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "feedback_dataset.csv")
MODEL_PATH = os.path.join(BASE, "models", "feedback_models.joblib")
TARGETS = ("sentiment", "emotion", "category")

SENTIMENT_OF = {
    "Joy": "Positive", "Surprise": "Positive", "Neutral": "Neutral",
    "Anger": "Negative", "Sadness": "Negative", "Fear": "Negative",
}

# category -> emotion -> example sentences ({item}/{n} are filled randomly)
TEMPLATES = {
    "Delivery": {
        "Anger": ["My order is delayed for {n} days and nobody tells me anything!",
                  "This is unacceptable, my {item} still hasn't been delivered after {n} weeks!"],
        "Sadness": ["I'm very disappointed, my {item} still hasn't arrived.",
                    "Sad that my {item} came so late, I needed it for an event."],
        "Fear": ["I'm worried my {item} got lost in transit, the tracking has not updated.",
                 "I'm scared the courier left my {item} at the wrong address."],
        "Joy": ["Delivery was super fast, my {item} arrived a day early. Thank you!",
                "Loved the quick delivery, the courier was so friendly!"],
        "Surprise": ["Wow, my {item} arrived much earlier than I expected!",
                     "I didn't expect the delivery to come this soon, what a surprise!"],
        "Neutral": ["I placed an order for a {item} yesterday and expect delivery next week.",
                    "The tracking page says my {item} is out for delivery."],
    },
    "Payment": {
        "Anger": ["You charged my card twice and nobody will fix it, this is outrageous!",
                  "My payment keeps failing and I'm furious that the money was still deducted!"],
        "Sadness": ["I'm disappointed that my payment failed again at checkout.",
                    "Sad to see the payment for my {item} was declined without any reason."],
        "Fear": ["I'm worried my card details are not safe, I saw a strange charge.",
                 "I'm afraid the payment for my {item} went to the wrong account."],
        "Joy": ["Payment was smooth and instant, loved the easy checkout!",
                "Great, the UPI payment went through in seconds. Happy with it!"],
        "Surprise": ["Wow, I got instant cashback on my payment, what a surprise!",
                     "I didn't expect such a smooth payment process, pleasantly surprised!"],
        "Neutral": ["I paid for my {item} using a credit card yesterday.",
                    "The payment page asks for a billing address before confirming."],
    },
    "Product Quality": {
        "Anger": ["The {item} is terrible quality and I'm really angry I paid for this!",
                  "Completely unacceptable, the {item} broke on the first day!"],
        "Sadness": ["The product quality is poor and I'm really disappointed.",
                    "I'm sad because the {item} looks nothing like the pictures."],
        "Fear": ["I'm worried the {item} is unsafe, it gets very hot while using.",
                 "I'm afraid the {item} is a fake, the quality feels suspicious."],
        "Joy": ["Amazing product, I really love the {item}!",
                "Excellent quality, the {item} works perfectly and I'm so happy."],
        "Surprise": ["Wow, the {item} is far better than I expected!",
                     "I didn't expect such great build quality, impressive!"],
        "Neutral": ["The {item} comes in a black colour with a one year warranty.",
                    "I received the {item} in a standard cardboard box."],
    },
    "Technical Problem": {
        "Anger": ["The app crashes every time I open it, this is ridiculous and I'm furious!",
                  "I can't log in for {n} days and support does nothing, so annoying!"],
        "Sadness": ["I'm disappointed that the app keeps freezing on my phone.",
                    "It's sad that the website never loads properly for me."],
        "Fear": ["I'm worried I lost all my data after the app update.",
                 "I'm afraid my account was hacked, I can't log in anymore."],
        "Joy": ["The new update fixed everything, the app runs great now!",
                "Love how fast and smooth the app is after the update!"],
        "Surprise": ["Wow, the app loaded instantly after the update, I'm surprised!",
                     "I didn't expect the bug to be fixed so quickly!"],
        "Neutral": ["The app asks me to update to the latest version.",
                    "I opened the settings page to change my password."],
    },
    "Pricing": {
        "Anger": ["The price is a total rip-off, I'm angry about this hidden charge!",
                  "You raised prices again without notice, this is unacceptable!"],
        "Sadness": ["I'm disappointed, the {item} is too expensive for me now.",
                    "Sad that I can no longer afford the subscription after the price hike."],
        "Fear": ["I'm worried the prices will keep going up every month.",
                 "I'm afraid I am being overcharged compared to other stores."],
        "Joy": ["Great pricing, the {item} was a wonderful deal, so happy!",
                "I love the discount, the price is very affordable."],
        "Surprise": ["Wow, the {item} is much cheaper than I expected!",
                     "I didn't expect a sale this big, what a pleasant surprise!"],
        "Neutral": ["The {item} costs about the same as it did last month.",
                    "The pricing page lists three different plans."],
    },
    "Customer Service": {
        "Anger": ["The support agent was rude and hung up on me, I'm furious!",
                  "Nobody replies to my complaints, this service is unacceptable!"],
        "Sadness": ["I'm disappointed that customer care never called me back.",
                    "Sad that the support team didn't care about my problem."],
        "Fear": ["I'm worried support will never resolve my issue in time.",
                 "I'm anxious because nobody from customer care answered my urgent message."],
        "Joy": ["The support team was wonderful, they solved my issue quickly. Thank you!",
                "Very helpful agent, I'm really happy with the service!"],
        "Surprise": ["Wow, customer care replied within minutes, I'm surprised!",
                     "I didn't expect such a friendly agent, what a pleasant surprise!"],
        "Neutral": ["I contacted customer care to ask about my order status.",
                    "The helpline is open from nine to six on weekdays."],
    },
    "Refund/Return": {
        "Anger": ["I still haven't received my refund after {n} weeks, this is outrageous!",
                  "They refuse to accept my return, I'm so angry about this!"],
        "Sadness": ["I haven't received my refund yet and I'm really upset.",
                    "Sad that the return of my {item} was rejected."],
        "Fear": ["I'm worried I'll never get my refund back.",
                 "I'm scared the return pickup won't happen and I'll lose my money."],
        "Joy": ["The refund was processed in a day, I'm so happy, thank you!",
                "Loved the easy return process, everything was smooth."],
        "Surprise": ["Wow, the refund reached my account faster than expected!",
                     "I didn't expect the return pickup to be this quick, surprised!"],
        "Neutral": ["I raised a return request for my {item} yesterday.",
                    "The refund policy says money returns in five to seven days."],
    },
}

ITEMS = ["phone", "laptop", "headphones", "jacket", "watch", "charger", "shoes", "tablet"]
NUMBERS = ["2", "3", "5", "10", "two", "ten"]
PREFIXES = ["", "", "Honestly, ", "Hi team, ", "To be clear, ", "Update: "]
SUFFIXES = ["", "", " Just sharing my feedback.", " That is my experience.", " Please note this."]


def build_demo_dataset(per_sentence: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = set()
    for category, by_emotion in TEMPLATES.items():
        for emotion, sentences in by_emotion.items():
            for sentence in sentences:
                for _ in range(per_sentence):
                    text = sentence.format(item=rng.choice(ITEMS), n=rng.choice(NUMBERS))
                    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
                    rows.add((text, SENTIMENT_OF[emotion], emotion, category))
    df = pd.DataFrame(sorted(rows), columns=["text", "sentiment", "emotion", "category"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=10)),
    ])


def train(verbose: bool = True) -> dict:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).dropna(subset=["text", *TARGETS])
    else:
        df = build_demo_dataset()
        df.to_csv(DATA_PATH, index=False)
        if verbose:
            print(f"Generated demo dataset with {len(df)} rows -> {DATA_PATH}")

    models = {}
    for target in TARGETS:
        x_train, x_test, y_train, y_test = train_test_split(
            df["text"], df[target], test_size=0.2, random_state=42, stratify=df[target])
        pipe = make_pipeline().fit(x_train, y_train)
        if verbose:
            print(f"\n=== {target.upper()} ===")
            print(classification_report(y_test, pipe.predict(x_test), zero_division=0))
        models[target] = make_pipeline().fit(df["text"], df[target])  # final model uses all data

    joblib.dump(models, MODEL_PATH)
    if verbose:
        print(f"Saved models -> {MODEL_PATH}")
    return models


if __name__ == "__main__":
    train()
