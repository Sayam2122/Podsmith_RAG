# Free Audio Analysis RAG System

A complete **open-source RAG (Retrieval-Augmented Generation)** system for analyzing audio transcripts with emotions, topics, and speaker information. Built entirely with free tools - no API keys required!

## 🌟 Features

- **🎙️ Audio Analysis Integration**: Process comprehensive audio analysis with emotions, topics, and speaker data
- **🔍 Semantic Search**: Find relevant segments using natural language queries
- **😊 Emotion-Aware**: Filter and search by detected emotions (neutral, joy, sadness, etc.)
- **🎯 Topic-Based Retrieval**: Search within specific discussion topics
- **💬 Interactive Chat**: Ask questions about your audio content
- **🆓 Completely Free**: Uses only open-source tools

## 🛠️ Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Vector Database** | ChromaDB | Store and search embeddings |
| **Embeddings** | SentenceTransformers | Convert text to vectors |
| **LLM** | Ollama (Llama3.2/Qwen2.5) | Generate responses |
| **Interface** | Streamlit + CLI | User interaction |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Ollama (Local LLM)
# Windows/Mac: Download from https://ollama.ai/download
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Setup Ollama Models

```bash
# Start Ollama server
ollama serve

# Pull a model (choose one):
ollama pull llama3.2:3b      # Balanced (3.8GB)
ollama pull qwen2.5:1.5b     # Fastest (1.5GB) 
ollama pull mistral:7b       # High quality (4.1GB)
```

### 3. Run the System

#### Command Line Interface (Recommended for testing)
```bash
python cli_interface.py
```

#### Web Interface (Better for interactive use)
```bash
# Install Streamlit
pip install streamlit

# Run web interface
streamlit run chat_interface.py
```

#### Python API
```python
from free_rag_system import FreeAudioRAG

# Initialize
rag = FreeAudioRAG(model_name="llama3.2:3b")

# Process your audio analysis
rag.process_audio_analysis("comprehensive_analysis_report.json")

# Ask questions
result = rag.chat("What segments have high confidence neutral emotions?")
print(result["response"])
```

## 📋 Usage Examples

### Basic Queries
```python
# Find emotional segments
rag.chat("What segments show joy or happiness?", 
         filters={"emotion": "joy"})

# Topic-based search  
rag.chat("What does the speaker say about Krishna?",
         filters={"topics": ["krishna & chanting"]})

# High-confidence segments
rag.chat("Find reliable transcription segments",
         filters={"min_confidence": 0.8})
```

### Advanced Filtering
```python
# Combine multiple filters
filters = {
    "emotion": "neutral",
    "min_confidence": 0.9,
    "speaker": "Speaker 1",
    "topics": ["krishna & chanting"]
}

result = rag.chat("Summarize the main points", filters)
```

### CLI Commands
```bash
# In CLI interface:
🎙️ > What are the symptoms of love mentioned?
🎙️ > filter emotion neutral
🎙️ > search high confidence segments  
🎙️ > krishna  # Predefined Krishna query
🎙️ > symptoms # Predefined symptoms query
```

## 📊 Your Data Structure

The system processes your `comprehensive_analysis_report.json` which contains:

```json
{
  "comprehensive_analysis": {
    "block_wise_analysis": [
      {
        "block_id": 1,
        "block_summary": {...},
        "original_segments": [
          {
            "segment_id": 8,
            "text": "And then there is another category.",
            "emotions": {
              "combined_emotion": {
                "emotion": "neutral",
                "confidence": 0.902637779712677  // High confidence!
              }
            }
          }
        ]
      }
    ]
  }
}
```

## 🎯 Example Queries for Your Data

Based on your audio analysis, try these queries:

1. **High-Confidence Neutral Segments**
   ```
   What segments have high confidence neutral emotions?
   ```

2. **Krishna Consciousness Discussion**
   ```
   What does the speaker say about Krishna consciousness and chanting?
   ```

3. **Spiritual Symptoms**
   ```
   What are the symptoms of love of God mentioned in the audio?
   ```

4. **Empiric Philosophers**
   ```
   Find segments discussing empiric philosophers and their limitations
   ```

5. **Emotional Analysis**
   ```
   Which parts of the audio show the strongest emotional responses?
   ```

## ⚙️ Configuration

### Model Selection
| Model | Size | Speed | Quality | RAM Needed |
|-------|------|-------|---------|------------|
| `qwen2.5:1.5b` | 1.5B | ⚡ Fastest | Good | 4GB |
| `llama3.2:3b` | 3B | 🚀 Fast | Excellent | 8GB |
| `mistral:7b` | 7B | 💪 Moderate | Excellent | 16GB |

### Search Filters
- **emotion**: `neutral`, `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`
- **min_confidence**: `0.0` to `1.0` (emotion detection confidence)
- **speaker**: `Speaker 1`, etc.
- **topics**: `krishna & chanting`, `preaching & preach`, `symptoms & freed`, `body & material`

## 🔧 System Requirements

### Minimum
- **RAM**: 8GB
- **Storage**: 5GB (for models)
- **Python**: 3.8+

### Recommended  
- **RAM**: 16GB
- **GPU**: Any NVIDIA GPU (optional, speeds up processing)
- **Storage**: 10GB

## 🚨 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Verify model is installed
ollama list
```

### Memory Issues
```bash
# Use smaller model
ollama pull qwen2.5:1.5b

# Or reduce context in code
rag = FreeAudioRAG(model_name="qwen2.5:1.5b")
```

### Import Errors
```bash
# Install missing packages
pip install sentence-transformers chromadb requests numpy

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Performance Tips

1. **Use faster models** for real-time chat: `qwen2.5:1.5b`
2. **Pre-filter data** before semantic search
3. **Batch process** multiple queries
4. **Use GPU** if available for embeddings

## 🔒 Privacy & Security

- ✅ **Completely Local**: All processing happens on your machine
- ✅ **No API Calls**: No data sent to external services  
- ✅ **Offline Capable**: Works without internet connection
- ✅ **No Tracking**: No telemetry or data collection

## 🆘 Support

If you encounter issues:

1. Check the [requirements.txt](requirements.txt) for dependencies
2. Verify Ollama is running: `ollama serve`
3. Ensure your model is installed: `ollama list`
4. Check Python version: `python --version` (requires 3.8+)

## 📄 License

This project uses open-source components:
- ChromaDB: Apache 2.0 License
- SentenceTransformers: Apache 2.0 License  
- Ollama: MIT License

Your audio analysis system is now fully searchable and interactive! 🎉
