"""AI-Based Customer Feedback & Emotion Analysis System (Flask).

Run:  python app.py      ->  http://127.0.0.1:5000
"""
import csv
import io
import os
import sqlite3
from datetime import datetime

import joblib
from flask import (Flask, Response, flash, g, jsonify, redirect,
                   render_template, request, url_for)

import train_model
from nlp_utils import clean_text  # noqa: F401  (needed to unpickle the models)
from priority import get_priority

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "feedback.db")
MAX_TEXT = 2000
MAX_UPLOAD_ROWS = 5000

EMOJI = {"Joy": "😊", "Anger": "😡", "Sadness": "😢", "Fear": "😨",
         "Surprise": "😲", "Neutral": "😐"}
CATEGORY_ICON = {"Delivery": "📦", "Payment": "💳", "Product Quality": "📱",
                 "Technical Problem": "🔧", "Pricing": "💰",
                 "Customer Service": "👨‍💼", "Refund/Return": "🔄"}

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
app.jinja_env.globals.update(EMOJI=EMOJI, CATEGORY_ICON=CATEGORY_ICON)

# ---------------------------------------------------------------- ML models
if not os.path.exists(train_model.MODEL_PATH):
    print("No trained model found - training one now...")
    train_model.train(verbose=False)
MODELS = joblib.load(train_model.MODEL_PATH)


def analyze_many(texts):
    """Predict sentiment, emotion, category and priority for a list of texts."""
    per_task = {}
    for task in train_model.TARGETS:
        pipe = MODELS[task]
        probs = pipe.predict_proba(texts)
        best = probs.argmax(axis=1)
        per_task[task] = [(pipe.classes_[i], float(probs[row, i]))
                          for row, i in enumerate(best)]
    results = []
    for n, text in enumerate(texts):
        sentiment, s_conf = per_task["sentiment"][n]
        emotion, e_conf = per_task["emotion"][n]
        category, c_conf = per_task["category"][n]
        priority, score = get_priority(sentiment, emotion, category, text)
        results.append({
            "text": text, "sentiment": sentiment, "emotion": emotion,
            "category": category, "priority": priority, "priority_score": score,
            "confidence": round((s_conf + e_conf + c_conf) / 3, 3),
            "confidence_detail": {"sentiment": round(s_conf, 3),
                                  "emotion": round(e_conf, 3),
                                  "category": round(c_conf, 3)},
        })
    return results


# ----------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, sentiment TEXT, emotion TEXT, category TEXT,
            priority TEXT, confidence REAL, created_at TEXT)""")


def save_results(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db = get_db()
    db.executemany(
        "INSERT INTO feedback (text, sentiment, emotion, category, priority, confidence, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        [(r["text"], r["sentiment"], r["emotion"], r["category"], r["priority"],
          r["confidence"], now) for r in results])
    db.commit()


def get_stats():
    db = get_db()

    def counts(column):  # column names are constants defined below, never user input
        rows = db.execute(f"SELECT {column}, COUNT(*) FROM feedback GROUP BY {column}").fetchall()
        return {r[0]: r[1] for r in rows}

    total = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    sentiment = counts("sentiment")
    pct = {k: round(100 * sentiment.get(k, 0) / total, 1) if total else 0
           for k in ("Positive", "Neutral", "Negative")}
    top = db.execute("SELECT category, COUNT(*) c FROM feedback WHERE sentiment='Negative' "
                     "GROUP BY category ORDER BY c DESC LIMIT 1").fetchone()
    return {
        "total": total,
        "sentiment": sentiment, "sentiment_pct": pct,
        "emotion": counts("emotion"), "category": counts("category"),
        "priority": counts("priority"),
        "high_priority": db.execute("SELECT COUNT(*) FROM feedback WHERE priority='High'").fetchone()[0],
        "top_complaint": top["category"] if top else "—",
    }


# ------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html", result=None, text="")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("feedback", "").strip()
    if len(text) < 5 or len(text) > MAX_TEXT:
        flash(f"Enter feedback between 5 and {MAX_TEXT} characters.", "error")
        return redirect(url_for("index"))
    result = analyze_many([text])[0]
    save_results([result])
    return render_template("index.html", result=result, text=text)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API.  Body: {"feedback": "...", "save": true}"""
    data = request.get_json(silent=True) or {}
    text = str(data.get("feedback", "")).strip()
    if len(text) < 5 or len(text) > MAX_TEXT:
        return jsonify(error=f"'feedback' must be 5-{MAX_TEXT} characters"), 400
    result = analyze_many([text])[0]
    if data.get("save", True):
        save_results([result])
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    """Bulk-analyze a CSV that has a 'feedback' column (or use the first column)."""
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Choose a .csv file.", "error")
        return redirect(url_for("dashboard"))
    reader = csv.DictReader(io.TextIOWrapper(file.stream, encoding="utf-8-sig"))
    column = next((f for f in (reader.fieldnames or []) if f.strip().lower() == "feedback"),
                  (reader.fieldnames or [None])[0])
    texts = []
    for row in reader:
        value = (row.get(column) or "").strip()
        if len(value) >= 5:
            texts.append(value[:MAX_TEXT])
        if len(texts) >= MAX_UPLOAD_ROWS:
            break
    if not texts:
        flash("No usable feedback rows found in that file.", "error")
    else:
        save_results(analyze_many(texts))
        flash(f"Analyzed {len(texts)} feedback entries.", "success")
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    priority = request.args.get("priority", "")
    sentiment = request.args.get("sentiment", "")
    query, params = "SELECT * FROM feedback WHERE 1=1", []
    if priority in ("High", "Medium", "Low"):
        query += " AND priority=?"
        params.append(priority)
    if sentiment in ("Positive", "Neutral", "Negative"):
        query += " AND sentiment=?"
        params.append(sentiment)
    rows = get_db().execute(query + " ORDER BY id DESC LIMIT 100", params).fetchall()
    return render_template("dashboard.html", stats=get_stats(), rows=rows,
                           priority=priority, sentiment=sentiment)


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/export.csv")
def export_csv():
    rows = get_db().execute("SELECT * FROM feedback ORDER BY id").fetchall()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(rows[0].keys() if rows else
                    ["id", "text", "sentiment", "emotion", "category", "priority", "confidence", "created_at"])
    writer.writerows([tuple(r) for r in rows])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=feedback_analysis.csv"})


@app.route("/delete/<int:item_id>", methods=["POST"])
def delete(item_id):
    db = get_db()
    db.execute("DELETE FROM feedback WHERE id=?", (item_id,))
    db.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/clear", methods=["POST"])
def clear():
    db = get_db()
    db.execute("DELETE FROM feedback")
    db.commit()
    flash("All feedback cleared.", "success")
    return redirect(url_for("dashboard"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)
