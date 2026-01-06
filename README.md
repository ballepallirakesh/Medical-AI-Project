🏥 Medical AI Project – Clinical Specialty Prediction & RAG System
__________________________________________________________________________________
📌 Project Overview

This project is an end-to-end Medical NLP system designed to automatically classify clinical notes into medical specialties and demonstrate a Retrieval-Augmented Generation (RAG) architecture for medical question answering.

The focus of this project is production-ready architecture, model comparison, and system design, not just raw accuracy.
__________________________________________________________________________________
##🎯 Problem Statement
-**Given a medical transcription (doctor notes, discharge summaries, clinical reports)**:
-1.Predict the medical specialty (e.g., Cardiology, Neurology, Surgery, etc.)
-2.Handle real-world challenges:
    - Severe class imbalance
    - Long, noisy medical text
    - Minority specialty recall
-3.Demonstrate how the system can be extended to RAG-based clinical reasoning
__________________________________________________________________________________
📂 Dataset
   -Source: Medical Transcriptions Dataset
   -Total samples: 4,999
   -Original classes: 40 medical specialties
   -Final classes: 36 (rare classes merged into Other)
Class imbalance example:
   -Surgery → 1100+ samples
   -Some specialties → < 10 samples
This imbalance is intentional and realistic, reflecting real hospital data.
__________________________________________________________________________________
## 2. Architecture Overview
The system follows a modular and extensible architecture:

- **Text Encoder**: BioBERT is used to generate contextual embeddings  specialized for biomedical text.
- **Imbalance Handling**: Focal Loss is applied during training to improve   learning on minority classes.
- **Retrieval-Augmented Generation (RAG)**:
  - Retrieves relevant medical context from an external knowledge base
  - Augments the model input without heavy retraining
- **Inference Layer**: Produces structured medical predictions from raw clinical text.

This design allows future upgrades without retraining the full model.
__________________________________________________________________________________
## 3. Folder Structure Explanation
Medical-AI-Project/
│
├── src/ # Core application code
│ ├── data/ # Data loading and preprocessing
│ ├── models/ # Model definitions
│ ├── training/ # Training and evaluation logic
│ └── inference/ # Prediction pipeline
│
├── data/ # Datasets (ignored in git)
├── artifacts/ # Training artifacts (ignored)
├── models/ # Saved model weights (ignored)
├── requirements.txt # Project dependencies
└── README.md
__________________________________________________________________________________
Folders containing sensitive data or large files are excluded using `.gitignore`.
__________________________________________________________________________________
## 4. Security Considerations
- No raw medical data is committed to version control
- Model weights and training artifacts are excluded from GitHub
- The project structure supports deployment in isolated environments
- No hardcoded credentials or secrets are used
- Designed for compliance with data privacy best practices
__________________________________________________________________________________
## 5. How to Run Locally
1. Clone the repository:
```bash
    git clone https://github.com/ballepallirakesh/Medical-AI-Project.git
    cd Medical-AI-Project
2. Create and activate virtual environment:
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Run training or inference modules from the src/ directory as required.
