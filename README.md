<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/React-19.1-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🏥 MedGen</h1>

<p align="center">
  <strong>AI-Powered Synthetic Medical Data Generation & Privacy Evaluation Platform</strong>
</p>

<p align="center">
  Generate privacy-preserving synthetic medical datasets using Large Language Models with built-in utility and privacy risk assessment.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Evaluation Pipeline](#-evaluation-pipeline)
- [Privacy Assessment](#-privacy-assessment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**MedGen** addresses a critical challenge in healthcare AI: the scarcity of accessible medical data due to privacy regulations (HIPAA, GDPR). By leveraging state-of-the-art Large Language Models with Retrieval-Augmented Generation (RAG), MedGen generates high-quality synthetic medical datasets that:

- ✅ Preserve statistical properties of original data
- ✅ Maintain utility for machine learning tasks
- ✅ Minimize privacy risks (singling out, linkability, inference attacks)
- ✅ Enable safe data sharing for research and development

---

## ✨ Features

### � Dataset Management System
- **Unified Dataset Hub**: Manage all datasets from a central location
- **Sample Datasets**: Pre-loaded medical datasets (Pima Diabetes, Diabetes Prediction, Andrew's Diabetes)
- **Save & Organize**: Save generated datasets with custom names and descriptions
- **One-Click Activation**: Instantly switch between datasets for analysis
- **Preview & Delete**: Preview any dataset or remove saved ones

### 🔬 Synthetic Data Generation
- **Dual Generation Modes**:
  - ⚡ **Fast Mode**: Single API call batch generation (~5-10 seconds for 10-50 rows)
  - 🧠 **Deep Mode**: Feature-by-feature RAG-enhanced generation (slower but more context-aware)
- **LLM-Powered Generation**: Uses GPT-4o-mini with customizable parameters
- **Auto-Batching**: Automatic batching for large requests (>25 rows)
- **Real-time Progress**: Live progress updates during generation
- **CSV Auto-Detection**: Automatic delimiter detection (comma, semicolon, tab, pipe)

### 📊 Data Analysis & Visualization
- **Interactive Data Explorer**: Upload, view, and analyze CSV datasets
- **Statistical Analysis**: Automatic computation of distributions, correlations, and summary statistics
- **Rich Visualizations**: Charts and graphs powered by Recharts

### 📥 Export & Download
- **Download Synthetic Data**: Export only the generated rows
- **Download Combined Data**: Export original + synthetic merged datasets
- **Save for Later**: Persist generated datasets for future use

### 🧪 Utility Evaluation
- **Multi-Model Comparison**: Evaluate with KNN, MLP, Naive Bayes, Random Forest, SGD, and SVM
- **Automated Pipeline**: Split → Train → Generate → Compare workflow
- **Performance Metrics**: Accuracy, precision, recall, F1-score, confusion matrices

### 🔒 Privacy Risk Assessment
- **Anonymeter Integration**: Industry-standard privacy risk metrics
- **Singling Out Risk**: Probability of uniquely identifying individuals
- **Linkability Risk**: Risk of linking records across datasets
- **Inference Risk**: Risk of inferring sensitive attributes

### 🖥️ Modern Web Interface
- **Material-UI v7 Design**: Clean, responsive interface with cyberpunk dark theme
- **Sidebar Navigation**: Quick access to all features
- **Real-time Updates**: Live generation progress and status
- **Natural Language Queries**: Ask questions about your data in plain English

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React 19)                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   Home   │ │ Datasets │ │ Explorer │ │ Analysis │ │ Generate │ ...      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Backend (Flask API)                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │    Dataset     │  │    Generate    │  │   Evaluation Pipeline      │    │
│  │   Management   │  │    Service     │  │   (ML Models + Privacy)    │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
│          │                   │                        │                     │
│  ┌───────▼───────────────────▼────────────────────────▼─────────────────┐  │
│  │                     Data Storage Layer                                │  │
│  │  ./data/saved_datasets/  │  ./data/generated/  │  ./data/chroma_db/  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                      ┌──────────────┼──────────────┐
                      ▼              ▼              ▼
              ┌──────────┐   ┌──────────┐   ┌──────────────┐
              │ ChromaDB │   │  OpenAI  │   │  Anonymeter  │
              │ (Vector) │   │   API    │   │   (Privacy)  │
              └──────────┘   └──────────┘   └──────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core language |
| **Flask 3.1** | REST API server |
| **LlamaIndex** | RAG framework |
| **ChromaDB** | Vector database for embeddings |
| **OpenAI GPT-4o-mini** | Synthetic data generation |
| **scikit-learn** | ML model evaluation |
| **Anonymeter** | Privacy risk assessment |
| **Pandas/NumPy** | Data processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 19** | UI framework |
| **Material-UI v7** | Component library |
| **Recharts** | Data visualization |
| **Framer Motion** | Animations |
| **Axios** | HTTP client |
| **React Router v7** | Navigation |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18+ and npm
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DESU-CLUB/CS3264.git
   cd CS3264
   ```

2. **Set up Python environment**
   ```bash
   # Using uv (recommended)
   pip install uv
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   # Start backend (terminal 1)
   uv run python backend.py
   
   # Start frontend (terminal 2)
   cd frontend && npm start
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - Health Check: http://localhost:5000/health

### Using Makefile

```bash
make install    # Install all dependencies
make dev        # Run both backend and frontend
make backend    # Run backend only
make frontend   # Run frontend only
make clean      # Clean generated files
```

### Docker (Alternative)

```bash
docker-compose up --build
```

---

## 📖 Usage

### 1. Manage Datasets
Navigate to **Datasets** page to:
- View all available sample datasets
- Activate a dataset with one click
- Save generated data for later use
- Preview any dataset before activating

### 2. Upload Custom Dataset
Go to **Data Explorer** and upload your own CSV file. The platform automatically detects delimiters (comma, semicolon, tab).

### 3. Generate Synthetic Data
Go to **Data Generation** and configure:
- **Generation Mode**: Fast (batch) or Deep (feature-by-feature)
- **Number of samples**: How many synthetic rows to generate
- **Temperature** (0.1-2.0): Controls randomness
- **Top-P** (0.1-1.0): Nucleus sampling threshold
- **Frequency Penalty**: Reduces repetitive patterns
- **Max Tokens**: Maximum tokens per API call

After generation:
- **Download** as CSV (synthetic only or combined)
- **Use for Analysis** to switch to the generated data
- **Save for Later** to store in your dataset library

### 4. Analyze Results
Use **Analysis** page to:
- View statistical distributions
- Generate charts and visualizations
- Compare original vs synthetic data

### 5. Natural Language Queries
Use the **Query Interface** to ask questions about your data in plain English, powered by RAG.

---

## 📡 API Reference

### Dataset Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/datasets` | List all datasets (sample + saved) |
| `POST` | `/datasets/<id>/activate` | Activate a dataset for analysis |
| `POST` | `/datasets/save` | Save generated data as new dataset |
| `DELETE` | `/datasets/<id>` | Delete a saved dataset |
| `GET` | `/datasets/<id>/preview` | Preview dataset (first 100 rows) |

### Data Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate_data` | Start synthetic data generation |
| `GET` | `/generation_status` | Check generation progress |
| `GET` | `/get_generated_data` | Retrieve generated data |
| `GET` | `/download_data?type=<type>` | Download as CSV (synthetic/combined/original) |
| `POST` | `/use_generated_data` | Switch to generated data for analysis |

### File Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload CSV dataset |
| `GET` | `/check_csv_status` | Check if CSV is loaded |
| `POST` | `/delete_current_csv` | Remove current CSV |
| `GET` | `/sample_datasets` | List sample datasets |
| `POST` | `/use_sample_dataset` | Use a sample dataset |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/stats_query` | Get statistical analysis |
| `POST` | `/stream_analysis` | Stream analysis results |
| `POST` | `/query_csv` | Execute pandas query |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/data_availability` | Check available data |

### Example: Generate Data (Fast Mode)

```bash
curl -X POST http://localhost:5000/generate_data \
  -H "Content-Type: application/json" \
  -d '{
    "numSamples": 50,
    "temperature": 0.7,
    "topP": 0.9,
    "repetitionPenalty": 1.1,
    "maxTokens": 4096,
    "generationMode": "fast"
  }'
```

### Example: Save Generated Dataset

```bash
curl -X POST http://localhost:5000/datasets/save \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Study Data",
    "description": "100 synthetic diabetes records",
    "type": "combined"
  }'
```

---

## 🔬 Evaluation Pipeline

The evaluation pipeline (`basic_eval_pipeline.py`) performs:

1. **Data Splitting**: 80% training / 20% test
2. **Original Training**: Train 6 ML models on original training data
3. **Synthetic Generation**: Generate synthetic data matching training set size
4. **Synthetic Training**: Train same models on synthetic data
5. **Evaluation**: Compare both on the held-out test set
6. **Visualization**: Generate comparison plots and metrics

### Supported Models
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Naive Bayes
- Random Forest
- Stochastic Gradient Descent (SGD)
- Support Vector Machine (SVM)

### Run Evaluation
```bash
uv run python basic_eval_pipeline.py
```

### Multi-Dataset Evaluation
```bash
uv run python multi_dataset_pipeline.py
```

---

## 🔒 Privacy Assessment

MedGen uses [Anonymeter](https://github.com/statice/anonymeter) for privacy risk evaluation:

### Singling Out Risk
Measures the probability that a synthetic record can uniquely identify an individual from the original dataset.

### Linkability Risk
Assesses whether records in the synthetic dataset can be linked to records in external datasets.

### Inference Risk
Evaluates the risk of inferring sensitive attributes about individuals using the synthetic data.

### Run Privacy Evaluation
```bash
uv run python anonymeter_privacy_eval.py
```

---

## 📁 Project Structure

```
MedGen/
├── backend.py                 # Flask API server (main entry point)
├── generate_data.py           # LLM synthetic data generation (fast + deep modes)
├── rag.py                     # RAG system with ChromaDB
├── basic_eval_pipeline.py     # ML evaluation pipeline
├── multi_dataset_pipeline.py  # Multi-dataset evaluation
├── anonymeter_privacy_eval.py # Privacy risk assessment
├── preprocess.py              # Data preprocessing utilities
├── dquery.py                  # Feature analysis with LLM
│
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Home.js        # Landing page
│   │   │   ├── DatasetManager.js  # Dataset management UI
│   │   │   ├── DataExplorer.js    # Data upload and preview
│   │   │   ├── DataGeneration.js  # Generation interface
│   │   │   ├── Analysis.js        # Data analysis & charts
│   │   │   ├── Database.js        # Database info
│   │   │   ├── Sidebar.js         # Navigation sidebar
│   │   │   └── ...
│   │   ├── services/
│   │   │   └── api.js         # API client with all endpoints
│   │   └── App.js             # Main app with routing
│   └── package.json
│
├── data/                      # Runtime data storage
│   ├── saved_datasets/        # User-saved datasets
│   ├── generated/             # Generated synthetic data
│   ├── chroma_db/             # ChromaDB vector store
│   └── features/              # Feature documents for RAG
│
├── evals/                     # Evaluation module
│   ├── models/                # ML model implementations
│   │   ├── knn.py
│   │   ├── mlp.py
│   │   ├── naivebayes.py
│   │   ├── randomforest.py
│   │   ├── sgd.py
│   │   └── svm.py
│   ├── dataset/               # Evaluation datasets
│   └── pristine_datasets/     # Original unmodified datasets
│
├── datasets/                  # Sample datasets
├── results/                   # Generated results and plots
├── multi_dataset_results/     # Multi-dataset evaluation results
│
├── .vscode/                   # VS Code configuration
│   ├── launch.json            # Debug configurations
│   └── settings.json          # Editor settings
│
├── pyproject.toml             # Python project configuration (uv)
├── requirements.txt           # Python dependencies
├── Makefile                   # Build automation
├── docker-compose.yml         # Docker configuration
├── Dockerfile                 # Backend container
└── .env.example               # Environment template
```

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Workflow                               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Select Dataset │    │  Upload Custom  │    │   Use Sample    │
│   (Datasets)    │    │   (Explorer)    │    │   Dataset       │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   Active Dataset    │
                    │  (RAG Index Built)  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    Analyze      │   │    Generate     │   │     Query       │
│   (Analysis)    │   │  (Generation)   │   │   (Database)    │
└─────────────────┘   └────────┬────────┘   └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Generated Data     │
                    │ (Synthetic Rows)    │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    Download     │   │  Use for        │   │     Save        │
│    as CSV       │   │  Analysis       │   │   for Later     │
└─────────────────┘   └─────────────────┘   └────────┬────────┘
                                                      │
                                                      ▼
                                           ┌─────────────────────┐
                                           │   Saved Datasets    │
                                           │ (Datasets Library)  │
                                           └─────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built as part of **CS3264** coursework at the National University of Singapore
- Uses [Anonymeter](https://github.com/statice/anonymeter) for privacy evaluation
- Powered by [OpenAI](https://openai.com) GPT-4o-mini
- UI components from [Material-UI](https://mui.com)
- RAG framework by [LlamaIndex](https://www.llamaindex.ai/)

---

<p align="center">
  Made with ❤️ for privacy-preserving healthcare AI
</p>
