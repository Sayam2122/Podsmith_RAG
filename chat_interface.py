"""
Interactive Chat Interface for Free Audio RAG System
Web-based interface using Streamlit (free)
"""

import streamlit as st
import json
import time
from free_rag_system import FreeAudioRAG

# Configure Streamlit page
st.set_page_config(
    page_title="Free Audio Analysis RAG",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_loaded' not in st.session_state:
    st.session_state.analysis_loaded = False

def initialize_rag():
    """Initialize the RAG system"""
    with st.spinner("🔄 Initializing RAG system..."):
        try:
            rag = FreeAudioRAG(model_name="llama3.2:3b")
            
            # Check Ollama connection
            if not rag.check_ollama_connection():
                st.error("❌ Ollama not connected. Please start Ollama and install models.")
                st.code("""
# Install Ollama: https://ollama.ai/download
# Then run:
ollama serve
ollama pull llama3.2:3b  # or qwen2.5:1.5b for faster responses
                """)
                return None
            
            # Process audio analysis
            if st.session_state.analysis_loaded == False:
                with st.spinner("📊 Processing audio analysis data..."):
                    count = rag.process_audio_analysis("comprehensive_analysis_report.json")
                    st.success(f"✅ Processed {count} documents")
                    st.session_state.analysis_loaded = True
            
            return rag
        
        except Exception as e:
            st.error(f"❌ Error initializing RAG: {str(e)}")
            return None

def display_chat_message(message, is_user=True):
    """Display a chat message"""
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 70%;">
                <strong>You:</strong> {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background-color: #f1f1f1; color: black; padding: 10px; border-radius: 10px; max-width: 70%;">
                <strong>🤖 Assistant:</strong> {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("🎙️ Free Audio Analysis RAG System")
    st.markdown("*Powered by open-source: ChromaDB, SentenceTransformers, Ollama*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "LLM Model",
            ["llama3.2:3b", "qwen2.5:1.5b", "mistral:7b"],
            help="Choose your preferred model. qwen2.5:1.5b is fastest."
        )
        
        # Search filters
        st.subheader("🔍 Search Filters")
        
        emotion_filter = st.selectbox(
            "Emotion Filter",
            ["None", "neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
            help="Filter by detected emotion"
        )
        
        min_emotion_confidence = st.slider(
            "Min Emotion Confidence",
            0.0, 1.0, 0.5, 0.1,
            help="Minimum confidence for emotion detection"
        )
        
        speaker_filter = st.selectbox(
            "Speaker Filter",
            ["None", "Speaker 1"],
            help="Filter by speaker"
        )
        
        topic_filter = st.multiselect(
            "Topic Filter",
            ["krishna & chanting", "preaching & preach", "symptoms & freed", "body & material"],
            help="Filter by discussion topics"
        )
        
        max_results = st.slider("Max Results", 1, 10, 5)
        
        # Build filters dict
        filters = {}
        if emotion_filter != "None":
            filters["emotion"] = emotion_filter
        if min_emotion_confidence > 0:
            filters["min_confidence"] = min_emotion_confidence
        if speaker_filter != "None":
            filters["speaker"] = speaker_filter
        if topic_filter:
            filters["topics"] = topic_filter
        
        st.json(filters if filters else {"no_filters": True})
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag()
    
    if st.session_state.rag_system is None:
        st.stop()
    
    # Example queries
    st.subheader("💡 Example Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 High confidence neutral segments"):
            example_query = "What segments have high confidence neutral emotions?"
            st.session_state.current_query = example_query
        
        if st.button("🕉️ Krishna consciousness discussion"):
            example_query = "What does the speaker say about Krishna consciousness?"
            st.session_state.current_query = example_query
    
    with col2:
        if st.button("🤔 Empiric philosophers"):
            example_query = "Find segments where the speaker discusses empiric philosophers"
            st.session_state.current_query = example_query
        
        if st.button("❤️ Love of God symptoms"):
            example_query = "What are the symptoms of love of God mentioned?"
            st.session_state.current_query = example_query
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])
    
    # Chat input
    if 'current_query' in st.session_state:
        query = st.text_input("Ask about the audio analysis:", value=st.session_state.current_query)
        del st.session_state.current_query
    else:
        query = st.text_input("Ask about the audio analysis:", placeholder="e.g., What are the main topics discussed?")
    
    if st.button("Send") and query:
        # Add user message to history
        st.session_state.chat_history.append({
            "content": query,
            "is_user": True,
            "timestamp": time.time()
        })
        
        # Generate response
        with st.spinner("🤖 Generating response..."):
            try:
                result = st.session_state.rag_system.chat(query, filters, max_results)
                response = result["response"]
                sources = result["sources"]
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "content": response,
                    "is_user": False,
                    "timestamp": time.time(),
                    "sources": sources
                })
                
                # Display sources
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} segments)"):
                        for i, source in enumerate(sources, 1):
                            if 'segment_id' in source:
                                st.write(f"""
                                **{i}. Segment {source['segment_id']}**
                                - Block: {source['block_id']}
                                - Speaker: {source['speaker']}
                                - Time: {source['timestamp']}
                                - Emotion: {source['emotion']}
                                - Topics: {', '.join(source['topics'])}
                                """)
                            else:
                                st.write(f"""
                                **{i}. Block {source['block_id']} Summary**
                                - Topic: {source['topic']}
                                - Type: Summary
                                """)
            
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
        
        # Refresh the page to show new messages
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Analytics section
    st.subheader("📊 System Analytics")
    
    if st.session_state.analysis_loaded:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Processed", "✅ Loaded")
        
        with col2:
            st.metric("Chat Messages", len(st.session_state.chat_history))
        
        with col3:
            ollama_status = "🟢 Connected" if st.session_state.rag_system.check_ollama_connection() else "🔴 Disconnected"
            st.metric("Ollama Status", ollama_status)

if __name__ == "__main__":
    main()
