# 🌐 Data Analyst Agent – Web Scraping & QnA with LLM + DuckDB

A **FastAPI backend** for dynamically analyzing web content. This project scrapes web pages, builds vector embeddings for semantic search, and answers user questions using **OpenRouter LLMs**. It also generates visualizations for numeric data, returning all results in structured **JSON responses**, including base64-encoded images.

Perfect for researchers, analysts, and developers who need to **interact with web data intelligently**.

---

## ✨ Features

- **🌐Web Scraping**

  - Extracts clean text and structured tables from any public URL.
  - Ignores irrelevant elements like `<script>`, `<style>`, `<img>`, `<svg>`, and `<i>`.

- **🗄️DuckDB Integration**

  - Stores extracted structured data (tables, text chunks) and generated embeddings for semantic search
  - Enables fast analytical queries without requiring external databases
  - Embeddings allow the LLM to efficiently retrieve relevant context for each question

- **🧠 LLM Integration (OpenRouter)**

  - Answers natural language questions based on scraped content.
  - Supports dynamic queries on both text and tabular data.

- **📊 Automatic Visualization**

  - Detects numeric queries and generates scatterplots automatically.
  - Returns images encoded in base64 for seamless frontend display.

- **🌍 CORS & Frontend Ready**
  - Supports cross-origin requests for frameworks like React, Next.js, or Vue.

---

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or Poetry
- OpenRouter API key

### Installation

```bash
git clone https://github.com/sakthi-cannyfore/Data-Analyst-Agent-With-AI.git
cd Data-Analyst-Agent-With-AI
pip install -r requirements.txt
uvicorn app.main:app --reload


```

---

> Note: questions.txt file is required for all queries.

## Example Questions.txt file

```
https://en.wikipedia.org/wiki/2020_Summer_Olympics_medal_table

Which country ranked #1 in total medals?
How many countries won more than 10 gold medals?
Which country ranked #5 and how many total medals did they win?
draw a scatterplot of Gold vs Total medals.

```

## Get Your OpenRouter API Key

[OpenRouter](https://openrouter.ai/)

- Sign up and get your API key

---

## Example (Model/ApiKey)

```
Model = google/gemini-2.5-flash
Api_key = sk-or-v1-4ee651d6882ad61
```

## Testing the API with cURL

```bash

curl -X POST "http://127.0.0.1:8000/api" \
  -F "file=@uploaded_files/questions.txt" \
  -F "api_key=sk-xxxxxxx" \
  -F "model=google/gemini-2.5-flash" \
  -F "image_file=@path/to/your/image.png" \
  -F "csv_file=@path/to/your/file.csv"

```

## Example JSON response

```
["United States",
"10 countries won more than 10 gold medals.",
"ROC, 71",
"iVBORw0KGgoAAAANSUhEUgAAAeoAAAGGCAYAAAC0W8IbAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAA "
]

```

# License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
© 2025 Sakthi Murugesan
