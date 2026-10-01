# All Projects

A collection of portfolio, academic, learning, and experimental projects. The repository is organized by project maturity and purpose; most projects are independent and have their own dependencies and run instructions.

## Repository Layout

| Folder | Contents |
|---|---|
| [01-Portfolio](01-Portfolio/) | Selected applications and projects for portfolio presentation |
| [02-Academic-Projects](02-Academic-Projects/) | Coursework, database work, and computer science fundamentals |
| [03-Learning-Projects](03-Learning-Projects/) | Practice applications and projects built while learning |
| [04-Experiments](04-Experiments/) | Prototypes and exploratory work |
| [05-Archive](05-Archive/) | Older practice and archived material |
| [06-Utilities](06-Utilities/) | Standalone helper scripts |

## Portfolio

- [Advanced Port Scanner](01-Portfolio/Advance%20port%20scanner/) - Python desktop tool for authorized network assessment, with scan history and report export.
- [CropHealthAI](01-Portfolio/CropHealthAI/) - Streamlit image-classification app for crop leaf disease prediction using a pretrained Swin Transformer.
- [Linear Regression Project](01-Portfolio/linear_regression_project/) - Terminal-based house-price prediction workflow: train, evaluate, save, and use a linear regression model.
- [Professional IEEE Student Chapter Website](01-Portfolio/Professional_IEEE_StudentChapter_Website/) - Responsive static website for the IEEE VJIT Student Branch.
- [RAG-based AI](01-Portfolio/RAG-based-ai/) - Retrieval-augmented course assistant built from video transcripts, embeddings, and a local Ollama model. See its [README](01-Portfolio/RAG-based-ai/Readme.md) for setup and workflow.
- [URL Shortener](01-Portfolio/url-shortener/) - Express and MongoDB URL shortener with aliases, expiry, click tracking, and analytics.

## Academic Projects

- [Data Structures in C](02-Academic-Projects/data%20structures%20%20in%20c/) - Console To-Do List and two-player Tic-Tac-Toe demonstrating linked lists and game logic.
- [DBMS Project](02-Academic-Projects/DBMS%20Project/) - SQL schema and sample queries for an e-commerce-style database.
- [IITKML](02-Academic-Projects/IITKML/) - Machine-learning coursework and experiments, with datasets and notebooks/scripts.

## Learning Projects

- [Python Mini Projects](03-Learning-Projects/python-mini-projects/) - Small standalone tools: file organizer, password manager, PDF merger, quiz app, typing-speed test, and water reminder.
- [Quiz Wizard](03-Learning-Projects/quize%20wizard/) - Python terminal quiz game with categories, difficulty choices, and text-based question banks.
- [Quiz Wizard 2](03-Learning-Projects/quize%20wizard%202/) - CustomTkinter desktop quiz app with multiple categories, difficulty settings, explanations, and a saved leaderboard.
- [Student Task Manager](03-Learning-Projects/student-task-manager/) - Flask-based task management application; see its source and requirements for setup details.

## Experiments, Archive, and Utilities

- [Face Recognition Attendance System](04-Experiments/Face%20Recognition%20Attendance%20System/) - Experimental attendance application with face-recognition-related assets and web interface.
- [Practice](05-Archive/Practice/) - Archived sandbox and practice material.
- [read_docx.py](06-Utilities/read_docx.py) - Utility script for reading Word documents.

## Getting Started

There is no single dependency set or launch command for the whole repository. Open the folder for the project you want to run and follow its README or inspect its dependency manifest (`requirements.txt` or `package.json`). Projects without a setup guide may require configuring their own runtime and dependencies.

For Python projects, use a project-specific virtual environment where practical. Do not assume generated models, datasets, API credentials, or external services are available just because a project is present in the repository. In particular, the Advanced Port Scanner is intended for systems you own or are authorized to assess.
