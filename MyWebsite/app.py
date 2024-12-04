from flask import Flask, render_template, request, jsonify
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time
import os

app = Flask(__name__)

APIFY_URL = "https://api.apify.com/v2/acts/apidojo~tweet-scraper/run-sync-get-dataset-items?token=apify_api_IHuvX99Z05YokbYV5dOt9YU8DYwCKm4blir7"

# Dropbox link for model (direct download link)
DROPBOX_MODEL_URL = "https://www.dropbox.com/scl/fo/bi4edwijyht1s930835s9/AAFydGQNjCpUdHntch7Qj70?rlkey=lhdw3si95h7h308ns332rplfu&st=44l2hcor&dl=1"

MODEL_PATH = "./bert_model"


# Function to download the model from Dropbox
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("Downloading the model from Dropbox...")
    response = requests.get(DROPBOX_MODEL_URL)

    if response.status_code == 200:
        with open(f"{MODEL_PATH}/model.safetensors", "wb") as model_file:
            model_file.write(response.content)
        print("Model downloaded successfully!")
    else:
        print("Failed to download the model. Status code:", response.status_code)


# Check if the model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    download_model()

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(predictions, dim=1).item()

    # Sentiment logic (returning positive/negative sentiment)
    return "positive" if sentiment == 1 else "negative"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/platform")
def platform_page():
    return render_template("platform.html")


@app.route("/platform/<platform>")
def specific_platform(platform):
    if platform == "twitter":
        return render_template("twitter.html")
    elif platform in ["instagram", "facebook"]:
        return render_template("soon.html", platform=platform.capitalize())
    else:
        return "Platform not supported", 404


@app.route("/scrape_twitter", methods=["POST"])
def scrape_twitter():
    try:
        start_time = time.time()
        data = request.json
        search_term = data.get("hashtag")
        tweet_count = int(data.get("tweet_count"))

        if tweet_count > 30:
            return jsonify({"status": "error", "message": "You can only request up to 30 tweets at a time."}), 400

        print(f"Request received to fetch {tweet_count} tweets for search term '{search_term}'")

        if search_term.startswith("#"):
            search_term = search_term[1:]
            payload = {
                "searchTerms": [f"{search_term}"],
                "maxItems": tweet_count,
                "includeSearchTerms": False,
                "tweetLanguage": "ar",
                "sort": "Latest",
            }
        else:
            payload = {
                "searchTerms": [search_term],
                "maxItems": tweet_count,
                "includeSearchTerms": False,
                "onlyTwitterBlue": False,
                "onlyVerifiedUsers": True,
                "tweetLanguage": "ar",
                "sort": "Latest",
            }

        print("Sending request to Apify API...")
        response = requests.post(APIFY_URL, json=payload, timeout=240)
        response.raise_for_status()
        print(f"Apify API responded in {time.time() - start_time:.2f} seconds")

        tweets = response.json()
        print(f"Retrieved {len(tweets)} tweets")

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        tweet_texts = []
        analysis_start = time.time()
        for tweet in tweets:
            text = tweet.get("text", "No text available")
            text_without_hashtags = re.sub(r"#\w+", "", text).strip()
            text_without_links = re.sub(r"https?://\S+", "", text_without_hashtags).strip()
            sentiment = classify_sentiment(text_without_links)

            # Calculate sentiment percentages
            positive_percentage = sentiment_counts["positive"] * 100 / len(tweets)
            negative_percentage = sentiment_counts["negative"] * 100 / len(tweets)

            # Classify as neutral if both percentages are between 40-60%
            if 40 <= positive_percentage <= 60 and 40 <= negative_percentage <= 60:
                sentiment = "neutral"
                sentiment_counts["neutral"] += 1
            else:
                sentiment_counts[sentiment] += 1

            tweet_texts.append(text_without_links)
        print(f"Sentiment analysis completed in {time.time() - analysis_start:.2f} seconds")

        # Update pie chart logic to handle neutral sentiment
        fig, ax = plt.subplots()
        ax.pie(
            [sentiment_counts["positive"], sentiment_counts["negative"], sentiment_counts["neutral"]],
            labels=["Positive", "Negative", "Neutral"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#76c7c0", "#f76c6c", "#f3e04d"],  # Added color for neutral
        )
        ax.axis("equal")

        img = BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        pie_chart_base64 = base64.b64encode(img.getvalue()).decode("utf-8")
        img.close()

        return jsonify({"status": "success", "tweets": tweet_texts, "chart": pie_chart_base64})

    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return jsonify({"status": "error", "message": "Failed to fetch tweets from API"}), 500
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
