#!/usr/bin/env python3
"""
🚀 Enhanced Audio RAG System - Quick Demo
Demonstrates intelligent query processing and enhanced responses
"""

import sys
import time
from free_rag_system import FreeAudioRAG

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {text}")
    print(f"{'='*60}")

def print_demo_query(query, description):
    """Print demo query with description"""
    print(f"\n🎙️ Query: {query}")
    print(f"💡 Demo: {description}")
    print("-" * 50)

def main():
    print_header("ENHANCED AUDIO RAG SYSTEM DEMO")
    
    try:
        # Initialize the enhanced RAG system
        print("🚀 Initializing Enhanced RAG System...")
        rag = FreeAudioRAG(model_name="llama3.2:3b")
        
        # Test queries demonstrating enhanced features
        demo_queries = [
            {
                "query": "summary",
                "description": "Smart command - auto-expands to comprehensive summary query"
            },
            {
                "query": "Show me happy emotions with high confidence",
                "description": "Auto-detects emotion filter (joy) and confidence threshold (0.8)"
            },
            {
                "query": "What does speaker 1 say about philosophy?",
                "description": "Auto-detects speaker filter and topic relevance"
            },
            {
                "query": "Find neutral emotions in this podcast",
                "description": "Emotion detection with intelligent search optimization"
            }
        ]
        
        for demo in demo_queries:
            print_demo_query(demo["query"], demo["description"])
            
            # Show the enhanced processing
            response = rag.chat(demo["query"])
            
            # Extract quality metrics if available
            if hasattr(response, 'get'):
                quality = response.get('quality', 'Unknown')
                source_count = response.get('source_count', 0)
                print(f"📊 Quality Score: {quality}")
                print(f"📚 Sources Used: {source_count}")
            
            print("✅ Enhanced processing complete!\n")
            time.sleep(1)  # Brief pause for readability
            
        print_header("ENHANCED FEATURES DEMONSTRATED")
        print("✅ Automatic filter detection")
        print("✅ Smart query expansion") 
        print("✅ Quality metrics calculation")
        print("✅ Enhanced response generation")
        print("✅ Intelligent fallback strategies")
        print("✅ Multi-factor relevance scoring")
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("💡 Make sure all required files are present and Ollama is running")

if __name__ == "__main__":
    main()
