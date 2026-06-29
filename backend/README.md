# TEAM-5 Speech Analysis Pipeline

An advanced AI-powered speech analysis system that records audio, transcribes speech, analyzes communication patterns, and generates personalized feedback reports using Google Gemini API and RAG (Retrieval Augmented Generation).

## Features

- 🎙️ **Audio Recording**: Record audio directly from your microphone
- 🔧 **Audio Preprocessing**: Automatic noise reduction and normalization
- 📝 **Speech-to-Text**: Accurate transcription using Faster-Whisper
- 📊 **Speech Analysis**: Comprehensive acoustic feature extraction (pitch, energy, pauses, etc.)
- 🧠 **AI Agents**: Specialized agents for communication, confidence, and personality analysis
- 🔍 **RAG System**: Knowledge-enhanced reports using vector database retrieval
- 🛡️ **Guardrails**: Input/output validation and safety checks
- ✨ **Final Report**: AI-generated personalized feedback and recommendations

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB recommended
- **Storage**: ~5GB for models and dependencies
- **Microphone**: Required for audio recording

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GSMPRANEETH/TEAM-5.git
cd TEAM-5
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Google Gemini API (Required for LLM)

The project uses Google Gemini API for AI inference. The free tier requires no credit card.

1. Get your API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Configure it:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Verify configuration:
```bash
python -c "from llm1.llm_config import GEMINI_API_KEY; print('✅ Key configured' if GEMINI_API_KEY else '❌ Key not set')"

### 6. (Optional) Install GuardrailsAI for Enhanced Safety

GuardrailsAI provides additional input/output validation:

```bash
pip install guardrails-ai
guardrails configure
```

Install Hub validators (optional but recommended):

```bash
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/profanity_free
guardrails hub install hub://guardrails/detect_pii
```

## Configuration

### LLM Configuration

Configured via environment variables in `.env` file:

```bash
GEMINI_API_KEY=your_key_here           # Required - get free at aistudio.google.com/apikey
GEMINI_MODEL_NAME=gemini-2.0-flash     # Model to use
LLM_TEMPERATURE=0.3                     # Creativity level (0.0-1.0)
LLM_MAX_TOKENS=1024                     # Maximum response length
```

### RAG Configuration

Edit `rag/config.py` to customize retrieval settings:

```python
CHROMA_PERSIST_DIR = "./chroma_db"  # Vector database storage
TOP_K_RESULTS = 3                    # Number of documents to retrieve
```

### Recording Settings

Edit the configuration in `main.py`:

```python
DURATION = 45        # Recording duration in seconds
SAMPLE_RATE = 16000  # Audio sample rate (required for Whisper)
CHANNELS = 1         # Mono audio
```

## Usage

### Basic Usage - Full Pipeline

Run the complete pipeline (record, transcribe, analyze, and report):

```bash
python main.py
```

This will:
1. Record audio for 45 seconds (speak into your microphone)
2. Preprocess and clean the audio
3. Transcribe speech to text
4. Analyze speech features
5. Run AI agents for analysis
6. Generate a personalized AI report

### Testing Individual Components

#### Test Speech-to-Text Only

```bash
# Requires clean_audio.wav file
python speech_to_text.py
```

#### Test Speech Analysis Only

```bash
# Requires clean_audio.wav file
python -c "from pipeline import get_pipeline_output; print(get_pipeline_output('clean_audio.wav'))"
```

#### Test Agent System

```bash
python agent.py
```

#### Test RAG System

```bash
python test_rag.py
```

#### Test LLM Connection

```bash
python test_llm_step5.py
```

### Processing Existing Audio

To analyze an existing audio file instead of recording:

1. Place your audio file as `raw_audio.wav` in the project directory
2. Comment out the `record_audio()` call in `main.py`
3. Run `python main.py`

Or use the pipeline module directly:

```python
from pipeline import get_pipeline_output

# Analyze your audio file
result = get_pipeline_output("your_audio.wav")
print(result)
```

## Project Structure

```
TEAM-5/
├── main.py                 # Main entry point - full pipeline
├── agent.py                # Agent orchestrator
├── speech_to_text.py       # Whisper transcription
├── speech_features.py      # Acoustic feature extraction
├── pipeline.py             # Pipeline utilities
├── llm_helper.py           # LLM loader helper
├── guardrails_config.py    # Input/output validation
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── agents/                # Specialized AI agents
│   ├── communication_agent.py
│   ├── confidence_agent.py
│   └── personality_agent.py
│
├── llm/                   # LLM wrapper (for agents)
│   └── local_llm.py
│
├── llm1/                  # LLM configuration & report generation
│   ├── llm_config.py
│   ├── local_llm.py
│   ├── prompt_templates.py
│   └── report_generator.py
│
├── rag/                   # RAG system
│   ├── config.py
│   ├── retriever.py
│   ├── knowledge_base.py
│   ├── rag_pipeline.py
│   └── documents/         # Knowledge base documents
│       ├── communication_rules.md
│       ├── confidence_psychology.md
│       └── personality_traits.md
│
└── utils/                 # Utilities
    └── parser.py          # JSON parsing utilities
```

## Dependencies

### Core Dependencies

- **torch** & **torchaudio**: Deep learning framework for audio processing
- **faster-whisper**: Efficient speech-to-text transcription
- **librosa**: Audio analysis and processing
- **sounddevice** & **soundfile**: Audio recording and I/O
- **opensmile**: Acoustic feature extraction
- **pyannote.audio**: Speaker diarization and audio analysis

### LLM & Agent Framework

- **langchain**: LLM orchestration framework
- **langchain-google-genai**: Google Gemini integration for LangChain
- **google-generativeai**: Google Gemini API SDK
- **python-dotenv**: Environment variable management

### RAG System

- **chromadb**: Vector database for semantic search
- **transformers**: HuggingFace models for embeddings
- **sentence-transformers**: Efficient sentence embeddings

### Validation & Safety

- **guardrails-ai**: Input/output validation framework
- **pydantic**: Data validation

See `requirements.txt` for complete dependency list with versions.

## Troubleshooting

### Gemini API Issues

**Error**: `GEMINI_API_KEY not set` or `Gemini API not available`

**Solution**:
1. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Copy `.env.example` to `.env` and add your key
3. Restart the backend server
4. Verify internet connection

The system will fall back to a stub LLM if the API is unavailable, which returns mock data for testing.

### Microphone Issues

**Error**: Audio recording fails

**Solution**:
1. Check microphone permissions for your terminal/Python
2. List available audio devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
3. Update device ID in recording code if needed

### ChromaDB Issues

**Error**: ChromaDB initialization fails

**Solution**:
1. Ensure ChromaDB is installed: `pip install chromadb`
2. Clear the database: `rm -rf chroma_db/`
3. The system will fall back to keyword-based retrieval if ChromaDB is unavailable

### Import Errors

**Error**: Module not found

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.8+)

### Memory Issues

**Error**: Out of memory during processing

**Solution**:
1. Use a smaller LLM model in `llm1/llm_config.py`
2. Reduce `MAX_TOKENS` in the configuration
3. Close other memory-intensive applications

## Advanced Configuration

### Using a Different Gemini Model

Edit the `.env` file:

```bash
# Available models:
GEMINI_MODEL_NAME=gemini-2.0-flash      # Fast and free (default)
# GEMINI_MODEL_NAME=gemini-1.5-pro      # More capable, higher limits
# GEMINI_MODEL_NAME=gemini-1.5-flash    # Balanced option
```

### Customizing Knowledge Base

Add your own documents to the RAG system:

1. Create markdown files in `rag/documents/`
2. Restart the system to re-index

### Adjusting Recording Duration

Edit `main.py`:

```python
DURATION = 60  # Record for 60 seconds instead of 45
```

## Performance Tips

1. **Use GPU acceleration**: If available, install PyTorch with CUDA support
2. **Optimize Whisper**: Use smaller model sizes for faster transcription
3. **Reduce context length**: Decrease `TOP_K_RESULTS` in RAG config for faster retrieval
4. **Pre-download models**: Download all models before first run to avoid delays

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add license information here]

## Acknowledgments

- **Faster-Whisper**: OpenAI Whisper implementation by Guillaume Klein
- **Google Gemini**: Cloud LLM API by Google
- **LangChain**: LLM application framework
- **ChromaDB**: Vector database for AI applications
- **GuardrailsAI**: LLM output validation framework

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact the development team

## Changelog

### Current Version
- Full speech analysis pipeline
- RAG-enhanced AI reporting
- Multi-agent analysis system
- GuardrailsAI integration
- Google Gemini API integration (cloud-ready deployment)
- Environment-based configuration via .env
