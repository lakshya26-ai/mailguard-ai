# 🛡️ MailGuard AI

**Your intelligent shield against unwanted email.**

A spam/not-spam classifier built with a genuinely trained scikit-learn model
(TF-IDF + Multinomial Naive Bayes). The trained model is exported to plain
numbers and executed **in the browser**, so the whole app is a static site you
can host for free on **GitHub Pages**. No server, no LLMs, no external AI APIs,
and no email text ever leaves the visitor's device.

**Live demo:** https://lakshya26-ai.github.io/mailguard-ai/

## Features

- Real ML inference (not keyword rules), identical to the Python model
- Explainable verdicts: the words that pushed the score towards spam or safe are highlighted
- Animated confidence ring, probability bars, scanning progress, count-up stats
- Distinctive "Paper & Ink" design: warm paper / forest-ink palette, serif headlines, mono labels, hard offset shadows, a marker-highlight headline and a rubber-stamp verdict (SPAM / CLEARED) that slams onto the result
- Animated 3D logo: a CSS-3D envelope whose flap opens and letter slides out while a shield seal floats above it. The hero logo also tilts with the cursor
- Light/dark theme (follows system), responsive down to mobile, reduced-motion support
- 28 one-click example emails (13 spam, 15 safe) in scrollable rows, plus a "Surprise me" button. Spam types: lottery, phishing, delivery fee, pharma, crypto, fake job, bank KYC, free gift, CEO fraud, romance, tech support, prize draw, bill threat. Safe types: meetings, receipts, interviews, lab/class notices, code review, travel, appointments, family, and more
- Scan history and stats stored privately in `localStorage`, clearable anytime
- Friendly empty, invalid-input and error states; `Ctrl/⌘ + Enter` to analyze

## Screenshots

Add yours to `docs/screenshots/` and link them here.

## Tech stack

Python · Pandas · scikit-learn (TF-IDF, MultinomialNB) · joblib · vanilla HTML/CSS/JS · GitHub Pages

## ML approach

1. `clean_text()` lowercases, masks URLs / emails / numbers, strips punctuation and stopwords
2. `TfidfVectorizer` (unigrams + bigrams, 5,000 features, `min_df=2`)
3. `MultinomialNB(alpha=0.3)` on a stratified 80/20 split
4. `export_for_web()` writes vocabulary, IDF weights, class priors and log-probabilities to `docs/model.js`
5. `docs/engine.js` re-implements cleaning, TF-IDF and Naive Bayes in JavaScript. Outputs match the Python model to within rounding (checked on sample texts).

## Dataset

- **SMS Spam Collection** (public, ~5k labeled messages, de-duplicated)
- **`data/real_world_examples.csv`**: 155 hand-written, representative email examples (77 spam, 78 legitimate) covering common scam families (lottery, advance-fee, phishing, KYC and bank alerts, fake invoices, delivery and customs fees, utility-bill threats, pharma, crypto, sextortion, romance, tech-support, gift-card/CEO fraud, fake jobs, loans, SEO/domain spam) and everyday legitimate mail (meetings, receipts, interviews, coursework and university notices, transaction alerts, OTPs, notifications, appointments, family). They are oversampled 4x in the training split.
- Drop any extra corpus (SpamAssassin, Enron, ...) into `data/extra_*.csv` with `label,message` columns (`spam`/`ham`) and re-run training. It is picked up automatically.

## Model performance (held-out 20%)

| Metric | Score |
|---|---|
| Accuracy | 98.40% |
| Precision (spam) | 98.50% |
| Recall (spam) | 89.73% |
| F1 (spam) | 93.91% |

Confusion matrix `[ham, spam]`: `[[916, 2], [15, 131]]`.
The test set is mostly SMS, so scores say little about long-form email. The
email examples are a small seed set; adding a large email corpus is the best next step.

## Project structure

```
mailguard-ai/
├── docs/                      # the website (served by GitHub Pages)
│   ├── index.html             # UI, styles, animations
│   ├── engine.js              # in-browser TF-IDF + Naive Bayes inference
│   ├── model.js               # exported trained model (generated)
│   └── .nojekyll
├── data/
│   ├── spam.csv
│   └── real_world_examples.csv
├── model/                     # pickles + metrics.json (Python-side)
├── src/
│   ├── train_model.py         # train, evaluate, export docs/model.js
│   └── predict.py             # Python inference / CLI reuse
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
python src/train_model.py            # optional: retrain + regenerate docs/model.js
python -m http.server -d docs 8000   # open http://localhost:8000
```

(`docs/index.html` also works by double-clicking it.)

**Known limitation:** invoice-style phishing is the weakest area. A fake "overdue invoice" sample scored only ~40% spam because it resembles real invoices, so it was left out of the demo chips. More email training data would fix this.

## Deploy on GitHub Pages

```bash
git init && git add . && git commit -m "MailGuard AI"
git branch -M main
git remote add origin https://github.com/<your-username>/mailguard-ai.git
git push -u origin main
```

Then in the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder `/docs` → Save.**
The site goes live at `https://<your-username>.github.io/mailguard-ai/` in about a minute.

## Future improvements

- Train on a large email corpus and compare Logistic Regression / Linear SVM with cross-validation
- "Report wrong verdict" feedback loop and an optional retraining pipeline
- Multi-language support, browser extension, GitHub Actions to retrain and redeploy
- Unit tests for `clean_text()` and a JS/Python parity test in CI

## Disclaimer

Educational portfolio project, not a production email security product.
