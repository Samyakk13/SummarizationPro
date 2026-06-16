# Text Summarizer — Pegasus Transformer

An abstractive text summarization web application built with a fine-tuned **Pegasus transformer model** and deployed as an interactive **Flask** web app. Users can paste long documents and generate concise, high-quality summaries with adjustable length.

---

## Overview

This project takes long-form text and condenses it into meaningful summaries using state-of-the-art NLP. Unlike extractive summarisation that simply picks existing sentences, this uses **abstractive summarisation** — the model generates new, human-like sentences that capture the core meaning of the source text.

```
Long document → Pegasus model → Concise abstractive summary
```

---

## Features

- Fine-tuned Pegasus transformer for abstractive summarisation
- Interactive Flask web interface with a clean UI
- Adjustable summary length selection
- Real-time text processing through the browser
- Custom tokenizer for domain-specific text handling

---

## Tech Stack

- **Python** — core language
- **Pegasus (HuggingFace Transformers)** — abstractive summarisation model
- **PyTorch** — deep learning backend
- **Flask** — web application framework
- **HTML / CSS** — frontend interface
- **SentencePiece** — tokenization

---

## Project Structure

```
SummarizationPro/
│
├── app.py                 # Flask application — routes and model inference
├── model/                 # Fine-tuned Pegasus model weights (loaded from HF Hub)
├── tokenizer/             # Custom tokenizer files
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── spiece.model
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── style.css          # Styling
│   └── images/            # UI assets
├── requirements.txt       # Dependencies
└── README.md
```

> **Note:** The fine-tuned model weights (~2.3GB) are hosted on the HuggingFace Hub rather than in this repository, following best practice of keeping code and model weights separate. The app loads the model automatically at runtime.

---

## How It Works

1. **Input** — the user pastes long-form text into the web interface and selects a summary length
2. **Tokenisation** — the input text is tokenised using the custom SentencePiece tokenizer
3. **Inference** — the fine-tuned Pegasus model generates an abstractive summary
4. **Output** — the generated summary is displayed back to the user in the browser

---

## Model

The summariser is built on the **Pegasus** architecture (Pre-training with Extracted Gap-sentences for Abstractive Summarization), a transformer model specifically designed for summarisation tasks. The model was fine-tuned on a custom dataset to improve summary quality for the target domain.

**Evaluation metrics (ROUGE):**

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.0187 |
| ROUGE-2 | 0.00036 |
| ROUGE-L | 0.0185 |

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Samyakk13/SummarizationPro.git
cd SummarizationPro
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
python app.py
```

**5. Open in your browser**
```
http://127.0.0.1:5000
```

---

## Key Concepts Demonstrated

- Fine-tuning a pre-trained transformer model for a specific NLP task
- Abstractive vs extractive summarisation
- Transfer learning with HuggingFace Transformers
- Model evaluation using ROUGE metrics
- Building a web interface for ML model inference
- End-to-end NLP application from model to user interface

---

## Future Improvements

- Deploy live on HuggingFace Spaces for public access
- Add support for document upload (PDF, DOCX)
- Improve ROUGE scores through additional fine-tuning epochs and a larger dataset
- Add batch summarisation for multiple documents
- Implement summary quality feedback to enable continuous improvement

---

## Author

**Samyak Sharma** — MSc Business Analytics, University of Birmingham

[LinkedIn](https://www.linkedin.com/in/samyaksharma13/) · [GitHub](https://github.com/Samyakk13)
