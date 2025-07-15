"""
Command Line Interface for Free Audio RAG System
Simple terminal-based interface for testing
"""

import json
import os
from free_rag_system import FreeAudioRAG

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                Free Audio Analysis RAG System                ║
    ║              Powered by Open Source Tools                    ║
    ║        ChromaDB + SentenceTransformers + Ollama             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def print_help():
    print("""
    Available Commands:
    
    🔍 Intelligent Search (Auto-detects filters):
      Just ask your question naturally! Examples:
      • "What is the summary of this podcast?"
      • "Find high confidence neutral emotions"
      • "What does the speaker say about Krishna?"
      • "Show me segments where someone is happy"
      • "Describe the philosophical discussions"
      
    🔍 Quick Commands:
      summary                          - Get overall summary
      emotions                         - Show emotional analysis  
      topics                          - Show main topics discussed
      speakers                        - Show speaker information
      
    ⚙️ System Commands:
      help                            - Show this help
      status                          - Check system status
      stats                           - Show database statistics
      clear                           - Clear screen
      quit / exit                     - Exit the program
      
    � Tips:
      • The system automatically detects emotions, speakers, confidence levels, and topics
      • Use natural language - no need for complex commands
      • Ask follow-up questions for more detail
    """)

def main():
    print_banner()
    print("Initializing system... Please wait.")
    
    # Initialize RAG system
    try:
        rag = FreeAudioRAG(model_name="llama3.2:3b")
        
        # Check Ollama
        if not rag.check_ollama_connection():
            print("❌ Ollama not available. Please install and start Ollama:")
            print("   1. Download: https://ollama.ai/download")
            print("   2. Run: ollama serve")
            print("   3. Install model: ollama pull llama3.2:3b")
            return
        
        # Process analysis data
        analysis_file = "comprehensive_analysis_report.json"
        if not os.path.exists(analysis_file):
            print(f"❌ Analysis file not found: {analysis_file}")
            return
        
        print("📊 Processing audio analysis data...")
        count = rag.process_audio_analysis(analysis_file)
        print(f"✅ Successfully processed {count} documents")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Command loop
    print("\n✨ System ready! Ask me anything about the audio content.")
    print("💡 Examples:")
    print("   • 'What is the summary of this podcast?'")
    print("   • 'Find segments about Krishna consciousness'")
    print("   • 'Show me happy emotions with high confidence'")
    
    while True:
        try:
            user_input = input("\n🎙️ > ").strip()
            
            if not user_input:
                continue
            
            # Parse commands
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
            
            elif user_input.lower() == 'status':
                status = "🟢 Connected" if rag.check_ollama_connection() else "🔴 Disconnected"
                print(f"Ollama Status: {status}")
                
                # Get system stats
                stats = rag.get_system_stats()
                if 'error' not in stats:
                    print(f"Database: {stats['total_documents']} documents loaded")
                    if stats['emotions']:
                        top_emotions = sorted(stats['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"Top emotions: {', '.join([f'{e}({c})' for e, c in top_emotions])}")
            
            elif user_input.lower() == 'stats':
                print("📊 Detailed Statistics:")
                stats = rag.get_system_stats()
                if 'error' not in stats:
                    print(f"   Total Documents: {stats['total_documents']}")
                    print(f"   Segments: {stats['document_types'].get('segment', 0)}")
                    print(f"   Block Summaries: {stats['document_types'].get('block_summary', 0)}")
                    print(f"   Speakers: {len(stats['speakers'])}")
                    print(f"   Topics Found: {len(stats['topics'])}")
                    print(f"   Emotions: {', '.join(stats['emotions'].keys())}")
                else:
                    print(f"❌ {stats['error']}")
            
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
            
            # Quick command shortcuts
            elif user_input.lower() == 'summary':
                user_input = "What is the overall summary of this audio content?"
            
            elif user_input.lower() == 'emotions':
                user_input = "What are the main emotions expressed in this audio?"
            
            elif user_input.lower() == 'topics':
                user_input = "What are the main topics discussed in this audio?"
            
            elif user_input.lower() == 'speakers':
                user_input = "Who are the speakers and what do they discuss?"
            
            # Handle any query with intelligent processing
            if user_input and not user_input.lower() in ['help', 'status', 'stats', 'clear']:
                result = rag.chat(user_input)
                
                if result['response'].startswith('Error:'):
                    print(f"❌ {result['response']}")
                else:
                    print(f"\n💬 {result['response']}")
                    
                    # Show auto-detected filters if any
                    if result.get('filters_used'):
                        print(f"\n🎯 Auto-detected filters: {result['filters_used']}")
                    
                    # Show sources
                    if result['sources']:
                        print(f"\n📚 Sources ({len(result['sources'])} segments):")
                        for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                            if 'segment_id' in source:
                                print(f"   {i}. Segment {source['segment_id']} ({source['timestamp']}) - {source['speaker']} - {source['emotion']}")
                            else:
                                print(f"   {i}. Block {source['block_id']} - {source['topic']} ({source.get('duration', 0):.1f}s)")
                        
                        if len(result['sources']) > 3:
                            print(f"   ... and {len(result['sources']) - 3} more sources")
                print(f"🔍 Searching: {query}")
                result = rag.chat(query, search_filters)
                print(f"\n🤖 {result['response']}")
                print(f"\n📚 Found {len(result['sources'])} relevant segments")
            
            elif user_input.lower() == 'krishna':
                query = "What does the speaker say about Krishna consciousness?"
                search_filters = {**filters, 'topics': ['krishna & chanting']}
                print(f"🔍 Searching: {query}")
                result = rag.chat(query, search_filters)
                print(f"\n🤖 {result['response']}")
                print(f"\n📚 Found {len(result['sources'])} relevant segments")
            
            elif user_input.lower() == 'symptoms':
                query = "What are the symptoms of love of God mentioned in the audio?"
                search_filters = {**filters, 'topics': ['symptoms & freed']}
                print(f"🔍 Searching: {query}")
                result = rag.chat(query, search_filters)
                print(f"\n🤖 {result['response']}")
                print(f"\n📚 Found {len(result['sources'])} relevant segments")
            
            elif user_input.lower().startswith('emotions '):
                emotion = user_input.split(' ', 1)[1]
                query = f"Find segments with {emotion} emotion"
                search_filters = {**filters, 'emotion': emotion}
                print(f"🔍 Searching: {query}")
                result = rag.chat(query, search_filters)
                print(f"\n🤖 {result['response']}")
                print(f"\n📚 Found {len(result['sources'])} relevant segments")
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:]  # Remove 'search '
                print(f"🔍 Searching: {query}")
                result = rag.chat(query, filters)
                print(f"\n🤖 {result['response']}")
                print(f"\n📚 Found {len(result['sources'])} relevant segments")
            
            # Direct question
            else:
                print(f"🔍 Searching: {user_input}")
                result = rag.chat(user_input, filters)
                print(f"\n🤖 {result['response']}")
                
                # Show sources
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} segments):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        if 'segment_id' in source:
                            print(f"   {i}. Segment {source['segment_id']} ({source['timestamp']}) - {source['emotion']}")
                        else:
                            print(f"   {i}. Block {source['block_id']} summary - {source['topic']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Type 'help' for available commands.")

if __name__ == "__main__":
    main()
