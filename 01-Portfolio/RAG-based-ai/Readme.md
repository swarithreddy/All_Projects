# RAG-based AI Course Assistant

A command-line retrieval-augmented generation (RAG) prototype that answers questions about a video course. It transcribes course audio into timestamped text chunks, embeds those chunks, retrieves the five most similar chunks for a question, and asks a local language model to answer with video and timestamp references.

## Pipeline

1. Put source course videos in a `videos/` folder inside this project. The video filenames are expected to contain a course number in the form `#<number>` and use ` ｜ ` (space, vertical bar, space) between the title and the remaining filename, for example `Course #1 ｜ Introduction.mp4`.
2. Create an `audios/` folder. Run `python video_to_mp3.py` to convert the videos to MP3. This step requires FFmpeg installed and available on `PATH`.
3. Create a `jsons/` folder. Run `python mp3_to_json.py` to transcribe the audio with Whisper `large-v2`. The current script sets the transcription language to Hindi and translates the result to English. It writes timestamped chunks to `jsons/`.
4. Start Ollama and make the `bge-m3` embedding model available. Run `python preprocess_json.py` to embed each transcript chunk and save the resulting dataframe to `embeddings.joblib`.
5. Make the `llama3.2` Ollama model available, then run `python process_incoming.py`. Enter a course-related question when prompted. The script retrieves the top five transcript chunks, writes the assembled prompt to `prompt.txt`, prints the generated answer, and saves it to `response.txt`.

The `audios/` and `jsons/` directories are not included by default; create them before running their corresponding scripts. Keep generated or source media files out of version control unless they are intended to be shared.

## Requirements

- Python 3
- FFmpeg, available on `PATH`
- Ollama running locally at `http://localhost:11434`
- Ollama models `bge-m3` for embeddings and `llama3.2` for generation
- Python packages used by the scripts: `openai-whisper`, `requests`, `pandas`, `numpy`, `scikit-learn`, and `joblib`

Install the Python packages in a virtual environment. For example:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install openai-whisper requests pandas numpy scikit-learn joblib
```

Install FFmpeg and Ollama separately using their official installation instructions. In Ollama, download the models with:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

## Run Commands

Run these commands from this project directory and in order:

```bash
python video_to_mp3.py
python mp3_to_json.py
python preprocess_json.py
python process_incoming.py
```

The Whisper model is downloaded by the Whisper package when first loaded, which can take time and disk space. Processing video and audio can also require substantial CPU/GPU resources.

## Project Files

| File | Purpose |
|---|---|
| `video_to_mp3.py` | Converts files in `videos/` to MP3 files in `audios/` using FFmpeg. |
| `mp3_to_json.py` | Transcribes and translates audio, saving timestamped transcript chunks in `jsons/`. |
| `preprocess_json.py` | Calls Ollama embeddings for transcript chunks and creates `embeddings.joblib`. |
| `process_incoming.py` | Embeds a question, retrieves similar chunks, prompts Ollama, and saves the response. |
| `prompt.txt` | Prompt generated for the most recent question. |
| `response.txt` | Answer generated for the most recent question. |
| `embeddings.joblib` | Serialized transcript dataframe used for retrieval. Regenerate it after changing the transcript data. |

## Notes

- The scripts use relative paths, so run them from this directory.
- `process_incoming.py` expects `embeddings.joblib` to exist before it starts.
- The embedding and generation scripts call Ollama's local HTTP API directly. They currently do not include explicit handling for a stopped Ollama service, missing models, or malformed responses.
- The current scripts process the entire directories rather than accepting command-line options.


