"""
Test the Enhanced RAG System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from free_rag_system import FreeAudioRAG

def test_enhanced_rag():
    print("🧪 Testing Enhanced RAG System")
    print("=" * 50)
    
    # Initialize RAG system
    try:
        rag = FreeAudioRAG(model_name="llama3.2:3b")
        
        # Check Ollama connection
        if not rag.check_ollama_connection():
            print("❌ Ollama not available")
            return
        
        # Process audio analysis
        analysis_file = "comprehensive_analysis_report.json"
        if not os.path.exists(analysis_file):
            print(f"❌ Analysis file not found: {analysis_file}")
            return
        
        print("📊 Processing audio analysis data...")
        count = rag.process_audio_analysis(analysis_file)
        print(f"✅ Loaded {count} documents")
        
        # Test enhanced queries
        test_queries = [
            "What is the summary of this podcast?",
            "Find segments about Krishna consciousness",
            "What emotions are expressed with high confidence?",
            "Who are the speakers and what do they discuss?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            result = rag.chat(query)
            
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"📊 Quality: {result.get('confidence_level', 'Unknown')}")
            print(f"📚 Sources: {len(result['sources'])} segments")
            
            if result.get('filters_used'):
                print(f"🎯 Auto-filters: {result['filters_used']}")
        
        print("\n🎉 Enhanced RAG system test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_rag()
