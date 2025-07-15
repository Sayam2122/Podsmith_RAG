"""
Enhanced Command Line Interface for Free Audio RAG System
Intelligent natural language processing with auto-filter detection
"""

import json
import os
from free_rag_system import FreeAudioRAG

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            🎧 Smart Audio Analysis RAG System 🤖            ║
    ║              Powered by Open Source Tools                    ║
    ║        ChromaDB + SentenceTransformers + Ollama             ║
    ║                    Intelligent & Auto-Filtering              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def print_help():
    print("""
    🧠 INTELLIGENT AUDIO RAG SYSTEM
    
    Just ask naturally! The system automatically detects:
    ✨ Emotions, speakers, confidence levels, topics, and more
    
    📝 Example Questions:
      • "What is the summary of this podcast?"
      • "Find high confidence neutral emotions"
      • "What does the speaker say about Krishna?"
      • "Show me segments where someone is happy"
      • "Describe the philosophical discussions"
      • "Who are the speakers and what do they discuss?"
      • "What emotions are expressed in this audio?"
    
    🚀 Quick Commands:
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
      
    💡 Smart Features:
      🎯 Auto-detects filters from your question
      🔍 Relaxed search if no results found
      📊 Shows relevant sources and metadata
      🤖 Context-aware responses
    """)

def main():
    print_banner()
    print("🚀 Initializing intelligent system... Please wait.")
    
    # Initialize RAG system
    try:
        rag = FreeAudioRAG(model_name="llama3.2:3b")
        
        # Check Ollama connection
        if not rag.check_ollama_connection():
            print("\n❌ Ollama not available. Please install and start Ollama:")
            print("   1. Download: https://ollama.ai/download")
            print("   2. Run: ollama serve")
            print("   3. Install model: ollama pull llama3.2:3b")
            return
        
        # Process audio analysis
        analysis_file = "comprehensive_analysis_report.json"
        if not os.path.exists(analysis_file):
            print(f"❌ Analysis file not found: {analysis_file}")
            return
        
        print("📊 Processing audio analysis data...")
        count = rag.process_audio_analysis(analysis_file)
        print(f"✅ Loaded {count} documents successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Command loop
    print("\n🎉 System ready! Ask me anything about the audio content.")
    print("💡 Try these examples:")
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
                print("👋 Thanks for using the Smart Audio RAG System!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
            
            elif user_input.lower() == 'status':
                print("🔍 System Status:")
                status = "🟢 Connected" if rag.check_ollama_connection() else "🔴 Disconnected"
                print(f"   Ollama: {status}")
                
                # Get system stats
                stats = rag.get_system_stats()
                if 'error' not in stats:
                    print(f"   Database: {stats['total_documents']} documents loaded")
                    if stats['emotions']:
                        top_emotions = sorted(stats['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top emotions: {', '.join([f'{e}({c})' for e, c in top_emotions])}")
                    print(f"   Speakers: {len(stats['speakers'])}")
                    print(f"   Topics: {len(stats['topics'])}")
                else:
                    print(f"   ❌ Database error: {stats['error']}")
            
            elif user_input.lower() == 'stats':
                print("📊 Detailed Statistics:")
                stats = rag.get_system_stats()
                if 'error' not in stats:
                    print(f"   📄 Total Documents: {stats['total_documents']}")
                    print(f"   🎭 Segments: {stats['document_types'].get('segment', 0)}")
                    print(f"   📋 Block Summaries: {stats['document_types'].get('block_summary', 0)}")
                    print(f"   🗣️  Speakers: {', '.join(stats['speakers'][:5])}")
                    if len(stats['speakers']) > 5:
                        print(f"       ... and {len(stats['speakers']) - 5} more")
                    print(f"   🏷️  Topics: {', '.join(stats['topics'][:8])}")
                    if len(stats['topics']) > 8:
                        print(f"       ... and {len(stats['topics']) - 8} more")
                    print(f"   😊 Emotions: {', '.join(stats['emotions'].keys())}")
                else:
                    print(f"   ❌ {stats['error']}")
            
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
            
            # Quick command shortcuts
            elif user_input.lower() == 'summary':
                user_input = "What is the overall summary of this audio content? Include main topics and key points."
            
            elif user_input.lower() == 'emotions':
                user_input = "What are the main emotions expressed in this audio? Show me the emotional patterns."
            
            elif user_input.lower() == 'topics':
                user_input = "What are the main topics and subjects discussed in this audio content?"
            
            elif user_input.lower() == 'speakers':
                user_input = "Who are the speakers in this audio and what do they discuss?"
            
            # Handle any query with intelligent processing
            if user_input and user_input.lower() not in ['help', 'status', 'stats', 'clear']:
                print("🧠 Processing your question intelligently...")
                result = rag.chat(user_input)
                
                if result['response'].startswith('Error:'):
                    print(f"❌ {result['response']}")
                else:
                    print(f"\n💬 {result['response']}")
                    
                    # Show quality metrics
                    quality = result.get('confidence_level', 'Unknown')
                    quality_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(quality, "⚪")
                    print(f"\n{quality_emoji} Response Quality: {quality}")
                    
                    # Show auto-detected filters if any
                    if result.get('filters_used'):
                        print(f"🎯 Smart filters detected: {result['filters_used']}")
                    
                    # Show processed query if different
                    if result.get('query_processed') and result['query_processed'] != user_input:
                        print(f"🔍 Optimized search: '{result['query_processed']}'")
                    
                    # Show sources with enhanced information
                    if result['sources']:
                        print(f"\n📚 Sources ({len(result['sources'])} segments):")
                        for i, source in enumerate(result['sources'][:4], 1):  # Show top 4
                            if 'segment_id' in source:
                                # Enhanced source display with confidence and emotion
                                confidence_emoji = "🎯" if source.get('confidence', 0) > 0.8 else "📍"
                                emotion_emoji = {
                                    "joy": "😊", "happy": "😊", "sad": "😢", "neutral": "😐", 
                                    "anger": "😠", "fear": "😨", "surprise": "😮", "disgust": "😤"
                                }.get(source.get('emotion', ''), "🎭")
                                
                                quality_indicator = ""
                                if source.get('confidence', 0) > 0.9:
                                    quality_indicator = " (High Quality)"
                                elif source.get('confidence', 0) > 0.7:
                                    quality_indicator = " (Good Quality)"
                                
                                print(f"   {confidence_emoji} Segment {source['segment_id']} ({source['timestamp']}) - {source['speaker']} {emotion_emoji}{quality_indicator}")
                                
                                if source.get('topics'):
                                    topics_str = ', '.join(source['topics'][:3])
                                    print(f"      🏷️  Topics: {topics_str}")
                            else:
                                print(f"   📋 Block {source['block_id']} - {source['topic']} ({source.get('duration', 0):.1f}s)")
                                if source.get('word_count'):
                                    print(f"      📝 Word count: {source['word_count']}")
                        
                        if len(result['sources']) > 4:
                            print(f"   ... and {len(result['sources']) - 4} more sources")
                    
                    # Show search performance with enhanced metrics
                    total_searched = result.get('search_results_count', 0)
                    if total_searched > 0:
                        quality_score = result.get('quality_score', 0)
                        print(f"\n⚡ Analysis: {total_searched} segments processed | Quality Score: {quality_score}/100")
                    
                    # Show suggestions if quality is low
                    if result.get('confidence_level') == 'Low' and result.get('suggestions'):
                        print(f"\n💡 Suggestions to improve results:")
                        for suggestion in result['suggestions'][:3]:
                            print(f"   • {suggestion}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Smart Audio RAG System!")
            break
        except Exception as e:
            print(f"❌ Oops! Something went wrong: {e}")
            print("💡 Try rephrasing your question or type 'help' for guidance.")

if __name__ == "__main__":
    main()
