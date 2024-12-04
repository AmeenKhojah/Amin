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

# Dropbox links for each model part
TOKENIZER_URL = "https://www.dropbox.com/scl/fi/24loqjtoxo058eyzhars3/tokenizer.json?rlkey=axwv7ys0acnhr3kviezrojr7w&st=785jq6gp&dl=1"
MODEL_TENSORS_URL = "https://www.dropbox.com/scl/fi/8gx6sqqba46tac9rwwjgs/model.safetensors?rlkey=ki58u0v4509rfggv6qc3av9k9&st=dfgr71kj&dl=1"
CONFIG_URL = "https://www.dropbox.com/scl/fi/jxcn1ap85r07y3rd7wdqe/config.json?rlkey=dkwoek62zft52xvanx81sv3fs&st=3kaxpjuf&dl=1"
SPECIAL_TOKENS_URL = "https://www.dropbox.com/scl/fi/uehztp14avyfhvmfobv3o/special_tokens_map.json?rlkey=2ocrxp72lmwc9l7xlr3ry5cqw&st=uy95odz9&dl=1"
TOKENIZER_CONFIG_URL = "https://www.dropbox.com/scl/fi/aczjnphx4tuw6qckk97lq/tokenizer_config.json?rlkey=sk6hguipw47oeq6ynii6s3ws4&st=7lhbhw10&dl=1"
TRAINING_ARGS_URL = "https://www.dropbox.com/scl/fi/egzlwrind6efz7zd0oiwd/training_args.bin?rlkey=nyc3x2ectrdabo64rjuosd57q&st=qozrhhm4&dl=1"
VOCAB_URL = "https://www.dropbox.com/scl/fi/6vl79blwbq5c6m92jbtdu/vocab.txt?rlkey=epgrx4upfxycoombzwerh52ju&st=uv28w69i&dl=1"

# Absolute path to model directory
MODEL_PATH = os.path.join(os.getcwd(), "bert_model")  # Absolute path to avoid ambiguity

# Function to download model parts
def download_model_parts():
    if not os.path.exists(MODEL_PATH):
        try:
            os.makedirs(MODEL_PATH)
            print(f"Directory {MODEL_PATH} created successfully!")
        except Exception as e:
            print(f"Failed to create directory {MODEL_PATH}. Error: {e}")
            return  # Exit the function if directory creation fails

    # List of URLs and corresponding filenames
    model_files = [
        (TOKENIZER_URL, "tokenizer.json"),
        (MODEL_TENSORS_URL, "model.safetensors"),
        (CONFIG_URL, "config.json"),
        (SPECIAL_TOKENS_URL, "special_tokens_map.json"),
        (TOKENIZER_CONFIG_URL, "tokenizer_config.json"),
        (TRAINING_ARGS_URL, "training_args.bin"),
        (VOCAB_URL, "vocab.txt")
    ]

    for url, filename in model_files:
        file_path = os.path.join(MODEL_PATH, filename)
        
        # Only download if file does not exist
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"{filename} downloaded successfully!")
            else:
                print(f"Failed to download {filename}. Status code: {response.status_code}")
        else:
            print(f"{filename} already exists, skipping download.")

# Check if the model parts exist, if not, download them
required_files = ["tokenizer.json", "model.safetensors", "config.json", 
                  "special_tokens_map.json", "tokenizer_config.json", 
                  "training_args.bin", "vocab.txt"]

# Check for file existence in the absolute path
if not os.path.exists(MODEL_PATH) or not all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in required_files):
    download_model_parts()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to classify sentiment
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
        print(f"Error while calling Apify API: {e}")
        return jsonify({"status": "error", "message": "An error occurred while processing your request."}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)
