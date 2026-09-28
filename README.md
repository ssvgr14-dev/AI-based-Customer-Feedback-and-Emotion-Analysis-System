# AI-Based Customer Feedback & Emotion Analysis System

Flask app that reads customer feedback and predicts **sentiment**, **emotion**, **category** and **priority**, then shows the results on a dashboard.

## Pipeline
Feedback -> text cleaning -> TF-IDF (1-2 grams) -> 3 x Logistic Regression -> priority rules -> SQLite -> dashboard

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py           # optional: app.py trains automatically if no model exists
python app.py                   # open http://127.0.0.1:5000
```

## Pages and API
| Route | Purpose |
|---|---|
| `/` | Analyze one piece of feedback |
| `/dashboard` | KPIs, charts, filterable log, CSV upload/export |
| `POST /api/analyze` | JSON: `{"feedback": "...", "save": true}` |
| `GET /api/stats` | Aggregated statistics as JSON |
| `POST /upload` | Bulk analyze a CSV with a `feedback` column |
| `/export.csv` | Download all analyzed feedback |

```bash
curl -X POST http://127.0.0.1:5000/api/analyze -H "Content-Type: application/json" \
     -d '{"feedback": "My order is delayed for two weeks!"}'
```

## Files
- `app.py` - routes, prediction, SQLite storage, stats
- `train_model.py` - dataset generation and model training
- `nlp_utils.py` - text cleaning
- `priority.py` - priority scoring rules (edit weights here)
- `templates/`, `static/` - UI
- `data/feedback_dataset.csv` - created on first training; `data/sample_upload.csv` - test file for bulk upload

## Important: use real data before presenting accuracy
The bundled dataset is generated from templates so the project runs out of the box. Because the test split shares templates with the training split, the reported 100% accuracy is **not** meaningful. Replace `data/feedback_dataset.csv` (columns: `text, sentiment, emotion, category`) with real labelled data, for example GoEmotions or Kaggle e-commerce review datasets, then run `python train_model.py` again. Expect accuracies of roughly 70-90% on real text.

Emotion labels: Joy, Anger, Sadness, Fear, Surprise, Neutral.
Categories: Delivery, Payment, Product Quality, Technical Problem, Pricing, Customer Service, Refund/Return.

## Ideas to extend
NLTK lemmatization, user login, a time-trend chart, email alerts for High priority items, swapping Logistic Regression for a transformer model.
