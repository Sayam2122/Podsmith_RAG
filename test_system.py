"""
Test script for Free RAG System
Verifies all components are working
"""

import sys
import os
import json

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import sentence_transformers
        print("✅ sentence_transformers imported successfully")
        
        import chromadb
        print("✅ chromadb imported successfully")
        
        import requests
        print("✅ requests imported successfully")
        
        import numpy
        print("✅ numpy imported successfully")
        
        import json
        print("✅ json imported successfully")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding_model():
    """Test the embedding model"""
    print("\n🤖 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding generation
        test_text = "This is a test sentence for embeddings."
        embedding = model.encode([test_text])
        
        print(f"✅ Embedding model loaded successfully")
        print(f"✅ Generated embedding shape: {embedding.shape}")
        
        return True
    
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

def test_vector_database():
    """Test ChromaDB"""
    print("\n💾 Testing vector database...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create test client
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create test collection
        collection = client.create_collection("test_collection")
        
        # Add test document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"test": "metadata"}],
            ids=["test_id"]
        )
        
        # Query test
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        print("✅ ChromaDB working successfully")
        print(f"✅ Query returned {len(results['documents'][0])} documents")
        
        # Cleanup
        client.delete_collection("test_collection")
        
        return True
    
    except Exception as e:
        print(f"❌ Vector database error: {e}")
        return False

def test_analysis_file():
    """Test if the analysis file exists and is valid"""
    print("\n📊 Testing analysis file...")
    
    analysis_file = "comprehensive_analysis_report.json"
    
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return False
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check structure
        if 'comprehensive_analysis' not in data:
            print("❌ Invalid analysis file structure")
            return False
        
        blocks = data['comprehensive_analysis'].get('block_wise_analysis', [])
        total_segments = 0
        
        for block in blocks:
            segments = block.get('original_segments', [])
            total_segments += len(segments)
        
        print(f"✅ Analysis file loaded successfully")
        print(f"✅ Found {len(blocks)} blocks with {total_segments} total segments")
        
        # Find your high-confidence neutral segment
        segment_8_found = False
        for block in blocks:
            for segment in block.get('original_segments', []):
                if segment.get('segment_id') == 8:
                    emotion_data = segment.get('emotions', {}).get('combined_emotion', {})
                    if emotion_data.get('confidence', 0) > 0.9:
                        print(f"✅ Found segment 8 with high confidence: {emotion_data.get('confidence'):.3f}")
                        segment_8_found = True
                        break
        
        if not segment_8_found:
            print("ℹ️  Segment 8 with high confidence not found (but file is valid)")
        
        return True
    
    except Exception as e:
        print(f"❌ Analysis file error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection (optional)"""
    print("\n🦙 Testing Ollama connection...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            print("✅ Ollama server is running")
            print(f"✅ Available models: {model_names}")
            
            # Check for recommended models
            recommended = ['llama3.2:3b', 'qwen2.5:1.5b', 'mistral:7b']
            available_recommended = [m for m in recommended if m in model_names]
            
            if available_recommended:
                print(f"✅ Recommended models available: {available_recommended}")
            else:
                print("⚠️  No recommended models found. Install with:")
                print("   ollama pull llama3.2:3b")
            
            return True
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("⚠️  Ollama server not running")
        print("   Start with: ollama serve")
        print("   Install models: ollama pull llama3.2:3b")
        return False
    
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Free RAG System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Embedding Model", test_embedding_model),
        ("Vector Database", test_vector_database),
        ("Analysis File", test_analysis_file),
        ("Ollama Connection", test_ollama_connection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 4:  # All except Ollama is acceptable
        print("\n🎉 System is ready! You can now run:")
        print("   python cli_interface.py")
        print("   streamlit run chat_interface.py")
    else:
        print("\n⚠️  Some critical components failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
