"""
Free RAG System for Audio Analysis
Uses open-source components: ChromaDB, SentenceTransformers, Ollama
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class FreeAudioRAG:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3.2:3b"):
        """
        Initialize the free RAG system
        
        Args:
            ollama_url: Ollama server URL
            model_name: LLM model to use (llama3.2:3b, qwen2.5:1.5b, mistral:7b)
        """
        # Initialize embedding model (free)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector database (free)
        print("Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path="./audio_analysis_vectordb",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Ollama configuration (free)
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.collection = None
        
        print(f"RAG system initialized with model: {model_name}")
    
    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """
        Flatten metadata to only contain primitive types (str, int, float, bool, None)
        ChromaDB doesn't accept nested dictionaries or lists as metadata values
        """
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Convert dict to JSON string
                flattened[key] = json.dumps(value)
            elif isinstance(value, list):
                # Convert list to comma-separated string for simple types
                if value and all(isinstance(item, (str, int, float, bool)) for item in value):
                    flattened[key] = ', '.join(str(item) for item in value)
                else:
                    # Convert complex list to JSON string
                    flattened[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                # Keep primitive types as-is
                flattened[key] = value
            else:
                # Convert other types to string
                flattened[key] = str(value)
        return flattened
    
    def _unflatten_metadata(self, metadata: Dict) -> Dict:
        """
        Unflatten metadata by parsing JSON strings back to objects where appropriate
        """
        unflattened = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Try to parse JSON strings back to objects
                if key in ['all_emotions'] and value.startswith(('[', '{')):
                    try:
                        unflattened[key] = json.loads(value)
                    except json.JSONDecodeError:
                        unflattened[key] = value
                elif key in ['topics', 'key_points'] and ',' in value:
                    # Handle comma-separated values
                    unflattened[key] = [item.strip() for item in value.split(',') if item.strip()]
                elif key in ['topics', 'key_points'] and value.startswith('['):
                    # Handle JSON arrays
                    try:
                        unflattened[key] = json.loads(value)
                    except json.JSONDecodeError:
                        unflattened[key] = [value] if value else []
                else:
                    unflattened[key] = value
            else:
                unflattened[key] = value
        
        # Ensure required fields have defaults
        unflattened.setdefault('type', 'segment')
        unflattened.setdefault('topic', 'Unknown Topic')
        unflattened.setdefault('topics', [])
        unflattened.setdefault('key_points', [])
        unflattened.setdefault('speaker', 'Unknown')
        unflattened.setdefault('emotion', 'unknown')
        unflattened.setdefault('start_time', 0)
        unflattened.setdefault('end_time', 0)
        unflattened.setdefault('emotion_confidence', 0)
        
        return unflattened
    
    def _extract_filters_from_query(self, query: str) -> tuple[str, Dict]:
        """
        Automatically extract filters and clean query based on content
        
        Returns:
            tuple: (cleaned_query, filters_dict)
        """
        import re
        
        filters = {}
        cleaned_query = query.lower()
        
        # Emotion detection
        emotions = ['joy', 'sad', 'anger', 'fear', 'surprise', 'disgust', 'neutral', 'happy', 'angry', 'excited', 'calm']
        for emotion in emotions:
            if emotion in cleaned_query:
                filters['emotion'] = emotion
                break
        
        # High confidence detection
        if any(phrase in cleaned_query for phrase in ['high confidence', 'confident', 'certain', 'sure']):
            filters['min_confidence'] = 0.8
        
        # Speaker detection (common names/terms)
        speaker_patterns = [
            r'\bspeaker\s+(\w+)',
            r'\b(krishna|guru|teacher|student|devotee|disciple)\b',
            r'\bby\s+(\w+)',
        ]
        for pattern in speaker_patterns:
            match = re.search(pattern, cleaned_query)
            if match:
                filters['speaker'] = match.group(1)
                break
        
        # Topic detection
        topic_keywords = {
            'krishna': ['krishna', 'chanting', 'devotion', 'bhakti'],
            'philosophy': ['philosophy', 'philosophical', 'empiric', 'empirical', 'logic'],
            'consciousness': ['consciousness', 'awareness', 'mind', 'soul'],
            'love': ['love', 'affection', 'attachment', 'symptoms'],
            'god': ['god', 'divine', 'supreme', 'absolute'],
            'spiritual': ['spiritual', 'transcendental', 'meditation'],
            'vedic': ['vedic', 'vedas', 'scripture', 'knowledge']
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in cleaned_query for keyword in keywords):
                detected_topics.append(topic)
        
        if detected_topics:
            filters['topics'] = detected_topics
        
        # Clean query by removing filter indicators
        cleaning_patterns = [
            r'\bhigh confidence\b',
            r'\bconfident\b',
            r'\bspeaker\s+\w+',
            r'\bby\s+\w+',
            r'\bwith\s+(high|low|good|bad)\s+(confidence|emotion)',
            r'\b(find|show|get)\s+',
            r'\bsegments?\s+(with|that|where)\b'
        ]
        
        for pattern in cleaning_patterns:
            cleaned_query = re.sub(pattern, '', cleaned_query)
        
        # Clean up extra spaces
        cleaned_query = ' '.join(cleaned_query.split())
        
        return query if not cleaned_query.strip() else cleaned_query, filters
    
    def _get_smart_response_context(self, query: str) -> str:
        """
        Generate context-aware instructions for better responses
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'main points']):
            return "Provide a comprehensive summary covering the main points, key topics, and overall themes."
        
        elif any(word in query_lower for word in ['emotion', 'feeling', 'sentiment', 'mood']):
            return "Focus on emotional aspects, speaker sentiments, and emotional patterns in your response."
        
        elif any(word in query_lower for word in ['speaker', 'who', 'said', 'person']):
            return "Identify speakers and attribute quotes accurately. Focus on who said what."
        
        elif any(word in query_lower for word in ['time', 'when', 'duration', 'timestamp']):
            return "Include specific timestamps and time references in your response."
        
        elif any(word in query_lower for word in ['topic', 'about', 'discuss', 'mention']):
            return "Focus on the specific topics and subjects discussed, providing detailed explanations."
        
        elif any(word in query_lower for word in ['how', 'why', 'explain', 'describe']):
            return "Provide detailed explanations with examples and reasoning from the audio content."
        
        else:
            return "Provide a detailed and informative response based on the audio content."
    
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name in model_names:
                    print(f"✓ Ollama connected, model {self.model_name} available")
                    return True
                else:
                    print(f"✗ Model {self.model_name} not found. Available: {model_names}")
                    print(f"Run: ollama pull {self.model_name}")
                    return False
            else:
                print("✗ Ollama server not responding")
                return False
        except Exception as e:
            print(f"✗ Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
    
    def process_audio_analysis(self, analysis_file_path: str):
        """
        Process the comprehensive audio analysis JSON file
        
        Args:
            analysis_file_path: Path to comprehensive_analysis_report.json
        """
        print(f"Loading analysis data from {analysis_file_path}...")
        
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Extract segments for processing
        segments = []
        block_summaries = []
        
        for block in analysis_data['comprehensive_analysis']['block_wise_analysis']:
            block_id = block['block_id']
            
            # Process block-level data
            if block.get('block_summary') and block.get('block_content'):
                block_summary = block['block_summary']
                block_content = block['block_content']
                
                block_summaries.append({
                    'id': f"block_{block_id}",
                    'type': 'block_summary',
                    'content': block_summary.get('summary', ''),
                    'metadata': {
                        'block_id': block_id,
                        'topic': block_summary.get('topic', ''),
                        'duration': block_content.get('duration', 0),
                        'word_count': block_content.get('word_count', 0),
                        'topics': block_content.get('topics', []),
                        'speaker': block_content.get('speaker', ''),
                        'key_points': block_summary.get('key_points', [])
                    }
                })
            
            # Process segment-level data
            for segment in block.get('original_segments', []):
                if segment.get('text'):  # Only process segments with text
                    emotion_data = segment.get('emotions', {}).get('combined_emotion', {})
                    
                    segments.append({
                        'id': f"segment_{segment['segment_id']}",
                        'type': 'segment',
                        'content': segment['text'],
                        'metadata': {
                            'segment_id': segment['segment_id'],
                            'block_id': block_id,
                            'speaker': segment.get('speaker', ''),
                            'start_time': segment.get('start', 0),
                            'end_time': segment.get('end', 0),
                            'confidence': segment.get('confidence', 0),
                            'emotion': emotion_data.get('emotion', 'unknown'),
                            'emotion_confidence': emotion_data.get('confidence', 0),
                            'all_emotions': emotion_data.get('all_emotions', {}),
                            'topics': block_content.get('topics', []) if 'block_content' in locals() else []
                        }
                    })
        
        print(f"Processed {len(segments)} segments and {len(block_summaries)} block summaries")
        
        # Create embeddings and store in vector database
        self._create_vector_database(segments + block_summaries)
        
        return len(segments) + len(block_summaries)
    
    def _create_vector_database(self, documents: List[Dict]):
        """Create embeddings and store in ChromaDB"""
        print("Creating embeddings...")
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection("audio_analysis")
        except:
            pass
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name="audio_analysis",
            metadata={"description": "Audio analysis with emotions and topics"}
        )
        
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Create rich context for embeddings
            contexts = []
            for doc in batch:
                if doc['type'] == 'segment':
                    # Handle topics as list or string
                    topics = doc['metadata']['topics']
                    topics_str = ', '.join(topics) if isinstance(topics, list) else str(topics)
                    
                    context = f"""
                    Text: {doc['content']}
                    Speaker: {doc['metadata']['speaker']}
                    Emotion: {doc['metadata']['emotion']} ({doc['metadata']['emotion_confidence']:.3f})
                    Topics: {topics_str}
                    Time: {doc['metadata']['start_time']:.1f}s-{doc['metadata']['end_time']:.1f}s
                    """
                else:  # block_summary
                    # Handle key_points as list or string
                    key_points = doc['metadata']['key_points']
                    if isinstance(key_points, list):
                        key_points_str = '; '.join(key_points[:3])
                    else:
                        key_points_str = str(key_points)
                    
                    context = f"""
                    Summary: {doc['content']}
                    Topic: {doc['metadata']['topic']}
                    Duration: {doc['metadata']['duration']:.1f}s
                    Key Points: {key_points_str}
                    """
                contexts.append(context.strip())
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(contexts, show_progress_bar=False)
            
            # Store in ChromaDB
            self.collection.add(
                ids=[doc['id'] for doc in batch],
                embeddings=embeddings.tolist(),
                documents=[doc['content'] for doc in batch],
                metadatas=[self._flatten_metadata(doc['metadata']) for doc in batch]
            )
            
            print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print("✓ Vector database created successfully")
    
    def search_segments(self, query: str, filters: Optional[Dict] = None, top_k: int = 8) -> Dict:
        """
        Enhanced search for relevant segments with better ranking
        
        Args:
            query: Search query
            filters: Optional filters (emotion, speaker, min_confidence, etc.)
            top_k: Number of results to return (increased default for better context)
        """
        if not self.collection:
            raise ValueError("No collection found. Run process_audio_analysis first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Build where clause for filtering
        where_clause = {}
        if filters:
            if 'emotion' in filters:
                where_clause['emotion'] = filters['emotion']
            if 'speaker' in filters:
                where_clause['speaker'] = filters['speaker']
            if 'min_confidence' in filters:
                where_clause['emotion_confidence'] = {"$gte": filters['min_confidence']}
            if 'min_speech_confidence' in filters:
                where_clause['confidence'] = {"$gte": filters['min_speech_confidence']}
        
        # Perform semantic search with higher result count for better filtering
        initial_results = top_k * 3  # Get more results for better ranking
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(initial_results, 50),  # Cap at 50 to avoid performance issues
            where=where_clause if where_clause else None
        )
        
        # Enhanced post-processing and ranking
        enhanced_results = self._enhance_search_results(query, results, filters, top_k)
        
        return enhanced_results
    
    def _enhance_search_results(self, query: str, results: Dict, filters: Optional[Dict], top_k: int) -> Dict:
        """
        Enhance search results with better ranking and filtering
        """
        if not results['documents'][0]:
            return results
        
        enhanced_items = []
        query_lower = query.lower()
        
        # Process each result for enhanced scoring
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Unflatten metadata
            unflattened_metadata = self._unflatten_metadata(metadata)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                query_lower, doc, unflattened_metadata, distance
            )
            
            # Apply topic filtering if specified
            if filters and 'topics' in filters:
                segment_topics = unflattened_metadata.get('topics', [])
                if isinstance(segment_topics, str):
                    segment_topics = [topic.strip() for topic in segment_topics.split(',') if topic.strip()]
                
                # Check if any filter topics match
                if not any(filter_topic.lower() in [t.lower() for t in segment_topics] for filter_topic in filters['topics']):
                    continue
            
            enhanced_items.append({
                'document': doc,
                'metadata': unflattened_metadata,
                'distance': distance,
                'relevance_score': relevance_score,
                'original_index': i
            })
        
        # Sort by relevance score (higher is better)
        enhanced_items.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Take top results and ensure diversity
        final_items = self._ensure_result_diversity(enhanced_items[:top_k * 2], top_k)
        
        # Reconstruct results format
        final_results = {
            'documents': [[item['document'] for item in final_items]],
            'metadatas': [[item['metadata'] for item in final_items]],
            'distances': [[item['distance'] for item in final_items]]
        }
        
        return final_results
    
    def _calculate_relevance_score(self, query_lower: str, document: str, metadata: Dict, distance: float) -> float:
        """
        Calculate enhanced relevance score based on multiple factors
        """
        score = 0.0
        doc_lower = document.lower()
        
        # Base semantic similarity (inverse of distance, normalized)
        semantic_score = max(0, 1 - distance) * 100
        score += semantic_score
        
        # Exact keyword matches in document (high weight)
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        exact_matches = len(query_words.intersection(doc_words))
        score += exact_matches * 15
        
        # Partial keyword matches
        for query_word in query_words:
            if len(query_word) > 3:  # Only for meaningful words
                for doc_word in doc_words:
                    if query_word in doc_word or doc_word in query_word:
                        score += 5
        
        # Topic relevance
        topics = metadata.get('topics', [])
        if isinstance(topics, list):
            topic_text = ' '.join(topics).lower()
            for query_word in query_words:
                if query_word in topic_text:
                    score += 10
        
        # Confidence boost for high-confidence segments
        emotion_confidence = metadata.get('emotion_confidence', 0)
        if emotion_confidence > 0.8:
            score += 5
        
        speech_confidence = metadata.get('confidence', 0)
        if speech_confidence > 0.8:
            score += 3
        
        # Document type preference (segments often more specific than summaries)
        if metadata.get('type') == 'segment':
            score += 2
        else:
            score += 8  # But summaries are good for overview questions
        
        # Length bonus for substantial content
        if len(document) > 100:
            score += 2
        
        return score
    
    def _ensure_result_diversity(self, items: List[Dict], target_count: int) -> List[Dict]:
        """
        Ensure diversity in results by avoiding too many similar segments
        """
        if len(items) <= target_count:
            return items
        
        diverse_items = []
        used_blocks = set()
        used_speakers = {}
        
        # First pass: take highest scoring items with diversity constraints
        for item in items:
            metadata = item['metadata']
            block_id = metadata.get('block_id', 'unknown')
            speaker = metadata.get('speaker', 'Unknown')
            
            # Limit segments per block and per speaker
            block_count = sum(1 for existing in diverse_items if existing['metadata'].get('block_id') == block_id)
            speaker_count = used_speakers.get(speaker, 0)
            
            # Allow up to 3 segments per block and 4 per speaker
            if block_count < 3 and speaker_count < 4:
                diverse_items.append(item)
                used_blocks.add(block_id)
                used_speakers[speaker] = speaker_count + 1
                
                if len(diverse_items) >= target_count:
                    break
        
        # Second pass: fill remaining slots if needed
        if len(diverse_items) < target_count:
            for item in items:
                if item not in diverse_items:
                    diverse_items.append(item)
                    if len(diverse_items) >= target_count:
                        break
        
        return diverse_items[:target_count]
    
    def generate_response(self, query: str, context_segments: Dict, max_tokens: int = 500) -> str:
        """
        Generate enhanced response using Ollama LLM with better prompting
        
        Args:
            query: User question
            context_segments: Retrieved segments from search
            max_tokens: Maximum tokens for response
        """
        if not self.check_ollama_connection():
            return "Error: Ollama not available. Please start Ollama and ensure the model is installed."
        
        # Build detailed context from retrieved segments
        context_parts = []
        segment_count = 0
        summary_count = 0
        
        for i, (doc, metadata) in enumerate(zip(context_segments['documents'][0], 
                                                context_segments['metadatas'][0])):
            try:
                if metadata.get('type') == 'segment':
                    segment_count += 1
                    # Safe access with defaults
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', 0)
                    speaker = metadata.get('speaker', 'Unknown')
                    emotion = metadata.get('emotion', 'unknown')
                    emotion_confidence = metadata.get('emotion_confidence', 0)
                    topics = metadata.get('topics', [])
                    
                    # Handle topics as list or string
                    if isinstance(topics, list):
                        topics_str = ', '.join(topics) if topics else 'General'
                    else:
                        topics_str = str(topics) if topics else 'General'
                    
                    # Add confidence indicator
                    confidence_level = "High" if emotion_confidence > 0.8 else "Medium" if emotion_confidence > 0.5 else "Low"
                    
                    context_parts.append(f"""
[SEGMENT {segment_count}] Time: {start_time:.1f}s-{end_time:.1f}s | Speaker: {speaker} | Emotion: {emotion} ({confidence_level} confidence: {emotion_confidence:.2f})
Topics: {topics_str}
Content: "{doc}"
                    """.strip())
                else:  # block summary
                    summary_count += 1
                    # Safe access with defaults
                    topic = metadata.get('topic', 'Unknown Topic')
                    duration = metadata.get('duration', 0)
                    block_id = metadata.get('block_id', 'unknown')
                    
                    context_parts.append(f"""
[BLOCK SUMMARY {summary_count}] Block ID: {block_id} | Duration: {duration:.1f}s | Main Topic: {topic}
Summary: "{doc}"
                    """.strip())
            except Exception as e:
                # If there's an error processing metadata, include the document anyway
                context_parts.append(f"""
[CONTENT {i+1}] "{doc}"
                """.strip())
                print(f"Warning: Error processing metadata for item {i+1}: {e}")
        
        context = "\n\n".join(context_parts)
        
        # Get smart response context and query analysis
        response_context = self._get_smart_response_context(query)
        query_type = self._analyze_query_type(query)
        
        # Create enhanced prompt based on query type
        if query_type == "summary":
            prompt = f"""You are an expert audio content analyst. Based on the following audio transcript segments and summaries, provide a comprehensive summary.

AUDIO CONTENT ANALYSIS:
{context}

USER QUESTION: "{query}"

ANALYSIS INSTRUCTIONS:
1. Synthesize information from all {segment_count} segments and {summary_count} block summaries
2. Identify the main themes, topics, and key points discussed
3. Note any significant emotional patterns or speaker dynamics
4. Provide a well-structured summary with clear sections
5. Include specific examples and quotes where relevant
6. Mention timeframes for important discussions

RESPONSE FORMAT:
- Start with a brief overview (2-3 sentences)
- Break down main topics with supporting details
- Include emotional context and speaker insights
- End with key takeaways or conclusions

Provide a comprehensive and accurate summary:"""

        elif query_type == "emotional":
            prompt = f"""You are an expert in emotional analysis of audio content. Analyze the emotional patterns and sentiments in the following audio segments.

AUDIO CONTENT WITH EMOTIONAL DATA:
{context}

USER QUESTION: "{query}"

EMOTIONAL ANALYSIS INSTRUCTIONS:
1. Examine the emotional data (emotion type, confidence levels) for each segment
2. Identify emotional patterns and changes throughout the audio
3. Correlate emotions with specific topics or content
4. Note any significant emotional peaks or transitions
5. Consider the speaker's emotional journey
6. Provide insights into what emotions reveal about the content

RESPONSE FORMAT:
- Overview of dominant emotions and patterns
- Specific examples with timestamps and confidence levels
- Analysis of emotional context and meaning
- Insights into speaker's emotional state and content impact

Provide detailed emotional analysis:"""

        elif query_type == "specific":
            prompt = f"""You are an expert content analyst. Find and analyze specific information from the audio content to answer the user's targeted question.

AUDIO CONTENT SEGMENTS:
{context}

USER QUESTION: "{query}"

SPECIFIC ANALYSIS INSTRUCTIONS:
1. Carefully examine each segment for information relevant to the question
2. Extract direct quotes and specific details that answer the question
3. Provide timestamps for referenced content
4. Include speaker attribution for important statements
5. Cross-reference information across segments for completeness
6. If information is partial or unclear, indicate limitations

RESPONSE FORMAT:
- Direct answer to the question with supporting evidence
- Relevant quotes with timestamps and speakers
- Additional context that enriches the answer
- Clear indication if information is incomplete

Provide a precise and well-supported answer:"""

        else:  # general or exploratory
            prompt = f"""You are an expert audio content analyst. Provide a comprehensive and insightful response based on the audio content provided.

AUDIO CONTENT ANALYSIS:
{context}

USER QUESTION: "{query}"

COMPREHENSIVE ANALYSIS INSTRUCTIONS:
1. Thoroughly analyze all provided segments and summaries
2. Address the question from multiple relevant angles
3. Include specific examples, quotes, and references
4. Provide context about speakers, emotions, and topics
5. Draw meaningful insights and connections
6. Structure the response logically and clearly

RESPONSE GUIDELINES:
- {response_context}
- Use specific evidence from the audio content
- Include timestamps and speaker references where relevant
- Provide both facts and analytical insights
- Ensure accuracy and avoid speculation beyond the provided content

Provide a detailed and insightful response:"""

        # Call Ollama API with enhanced settings
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more accurate responses
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "num_ctx": 4096,  # Larger context window
                        "stop": ["USER QUESTION:", "INSTRUCTIONS:", "ANALYSIS:"]
                    }
                },
                timeout=120  # Longer timeout for detailed responses
            )
            
            if response.status_code == 200:
                generated_text = response.json()["response"].strip()
                
                # Post-process the response for better formatting
                return self._post_process_response(generated_text, query_type, segment_count, summary_count)
            else:
                return f"Error generating response: HTTP {response.status_code}"
        
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the query to determine the best response approach"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'main points', 'what is this about']):
            return "summary"
        elif any(word in query_lower for word in ['emotion', 'feeling', 'sentiment', 'mood', 'emotional']):
            return "emotional"
        elif any(word in query_lower for word in ['what', 'who', 'when', 'where', 'how', 'why', 'find', 'show me']):
            return "specific"
        else:
            return "general"
    
    def _post_process_response(self, response: str, query_type: str, segment_count: int, summary_count: int) -> str:
        """Post-process the LLM response for better formatting and accuracy"""
        
        # Clean up common LLM artifacts
        response = response.replace("Based on the provided audio content,", "")
        response = response.replace("According to the audio segments,", "")
        response = response.strip()
        
        # Add context footer for transparency
        footer = f"\n\n📊 *Analysis based on {segment_count} audio segments and {summary_count} block summaries*"
        
        # Add query-specific enhancements
        if query_type == "summary" and len(response) > 200:
            # Ensure summary has good structure
            if not any(marker in response for marker in ['**', '##', '###', '•', '-', '1.', '2.']):
                # Add basic structure if missing
                lines = response.split('\n')
                if len(lines) > 3:
                    structured = []
                    for i, line in enumerate(lines):
                        if line.strip() and i < 3:
                            structured.append(f"**{line.strip()}**")
                        else:
                            structured.append(line)
                    response = '\n'.join(structured)
        
        return response + footer
    
    def chat(self, query: str, filters: Optional[Dict] = None, top_k: int = 8) -> Dict:
        """
        Enhanced chat pipeline with intelligent processing and better accuracy
        
        Args:
            query: User question
            filters: Optional manual filters (will be merged with auto-detected ones)
            top_k: Number of context segments to retrieve (increased for better context)
        
        Returns:
            Dict with enhanced response and metadata
        """
        print(f"🔍 Analyzing query: {query}")
        
        # Auto-detect filters from query
        cleaned_query, auto_filters = self._extract_filters_from_query(query)
        
        # Merge manual filters with auto-detected ones (manual takes precedence)
        final_filters = auto_filters.copy()
        if filters:
            final_filters.update(filters)
        
        # Use cleaned query for search if it's substantially different
        search_query = cleaned_query if len(cleaned_query) > len(query) * 0.5 else query
        
        if final_filters:
            print(f"🎯 Auto-detected filters: {final_filters}")
        
        print(f"🔍 Searching for: {search_query}")
        
        # Enhanced search for relevant segments
        search_results = self.search_segments(search_query, final_filters, top_k)
        
        if not search_results['documents'][0]:
            # Try again with relaxed filters if no results
            if final_filters:
                print("🔄 No results found, trying with relaxed filters...")
                relaxed_filters = {}
                # Keep only emotion and topic filters, remove confidence requirements
                if 'emotion' in final_filters:
                    relaxed_filters['emotion'] = final_filters['emotion']
                if 'topics' in final_filters and len(final_filters['topics']) > 1:
                    # Try with fewer topics
                    relaxed_filters['topics'] = final_filters['topics'][:2]
                
                search_results = self.search_segments(search_query, relaxed_filters, top_k * 2)
        
        # If still no results, try broader search
        if not search_results['documents'][0]:
            print("🔄 Expanding search scope...")
            broader_query = self._create_broader_query(query)
            search_results = self.search_segments(broader_query, None, top_k * 2)
        
        if not search_results['documents'][0]:
            return {
                "response": f"I couldn't find any relevant information for '{query}' in the audio analysis data. The audio content might not cover this topic, or you could try rephrasing your question with different keywords.",
                "sources": [],
                "filters_used": final_filters,
                "query_processed": search_query,
                "suggestions": self._generate_search_suggestions()
            }
        
        print(f"📄 Found {len(search_results['documents'][0])} relevant segments")
        
        # Generate enhanced response
        print("🤖 Generating detailed response...")
        response = self.generate_response(query, search_results, max_tokens=600)  # Increased token limit
        
        # Prepare enhanced source information
        sources = []
        for metadata in search_results['metadatas'][0]:
            try:
                if metadata.get('type') == 'segment':
                    sources.append({
                        "segment_id": metadata.get('segment_id', 'unknown'),
                        "block_id": metadata.get('block_id', 'unknown'),
                        "speaker": metadata.get('speaker', 'Unknown'),
                        "timestamp": f"{metadata.get('start_time', 0):.1f}s-{metadata.get('end_time', 0):.1f}s",
                        "emotion": metadata.get('emotion', 'unknown'),
                        "confidence": metadata.get('emotion_confidence', 0),
                        "speech_confidence": metadata.get('confidence', 0),
                        "topics": metadata.get('topics', []),
                        "type": "segment"
                    })
                else:
                    sources.append({
                        "block_id": metadata.get('block_id', 'unknown'),
                        "topic": metadata.get('topic', 'Unknown Topic'),
                        "type": "summary",
                        "duration": metadata.get('duration', 0),
                        "word_count": metadata.get('word_count', 0)
                    })
            except Exception as e:
                print(f"Warning: Error processing source metadata: {e}")
        
        # Calculate response quality metrics
        quality_metrics = self._calculate_response_quality(query, search_results, response)
        
        return {
            "response": response,
            "sources": sources,
            "filters_used": final_filters,
            "query_processed": search_query,
            "search_results_count": len(search_results['documents'][0]),
            "quality_score": quality_metrics["overall_score"],
            "confidence_level": quality_metrics["confidence_level"]
        }
    
    def _create_broader_query(self, query: str) -> str:
        """Create a broader search query if the original doesn't find results"""
        query_lower = query.lower()
        
        # Extract key nouns and important terms
        import re
        words = re.findall(r'\b\w{4,}\b', query_lower)  # Words with 4+ characters
        
        # Remove common stop words
        stop_words = {'what', 'does', 'this', 'that', 'they', 'them', 'their', 'with', 'from', 'about', 'were', 'been', 'have', 'said'}
        key_words = [w for w in words if w not in stop_words]
        
        if key_words:
            return ' '.join(key_words[:3])  # Take top 3 key words
        else:
            return query
    
    def _generate_search_suggestions(self) -> List[str]:
        """Generate helpful search suggestions when no results are found"""
        return [
            "Try asking about the main topics discussed",
            "Ask about emotions or speaker patterns",
            "Request a general summary of the content",
            "Search for specific speakers or time periods",
            "Use broader terms related to your topic"
        ]
    
    def _calculate_response_quality(self, query: str, search_results: Dict, response: str) -> Dict:
        """Calculate quality metrics for the response"""
        
        # Basic quality indicators
        result_count = len(search_results['documents'][0])
        response_length = len(response)
        
        # Calculate average confidence of sources
        avg_confidence = 0
        if search_results['metadatas'][0]:
            confidences = []
            for metadata in search_results['metadatas'][0]:
                emotion_conf = metadata.get('emotion_confidence', 0)
                speech_conf = metadata.get('speech_confidence', 0)
                confidences.append(max(emotion_conf, speech_conf))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Determine overall quality score
        quality_score = 0
        if result_count >= 5:
            quality_score += 30
        elif result_count >= 3:
            quality_score += 20
        else:
            quality_score += 10
        
        if response_length > 200:
            quality_score += 25
        elif response_length > 100:
            quality_score += 15
        
        if avg_confidence > 0.8:
            quality_score += 25
        elif avg_confidence > 0.6:
            quality_score += 15
        
        # Check if response addresses the query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['summary', 'overview']) and 'summary' in response.lower():
            quality_score += 20
        if any(word in query_lower for word in ['emotion', 'feeling']) and 'emotion' in response.lower():
            quality_score += 20
        
        # Determine confidence level
        if quality_score >= 80:
            confidence_level = "High"
        elif quality_score >= 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            "overall_score": min(quality_score, 100),
            "confidence_level": confidence_level,
            "source_count": result_count,
            "avg_source_confidence": avg_confidence
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        if not self.collection:
            return {"error": "No collection found"}
        
        try:
            # Get collection info
            count = self.collection.count()
            
            # Try to get some sample metadata to analyze
            sample_results = self.collection.query(
                query_texts=["sample"],
                n_results=min(100, count)
            )
            
            # Analyze metadata
            emotions = {}
            speakers = set()
            topics = set()
            types = {"segment": 0, "block_summary": 0}
            
            for metadata in sample_results['metadatas'][0]:
                unflattened = self._unflatten_metadata(metadata)
                
                # Count emotions
                emotion = unflattened.get('emotion', 'unknown')
                emotions[emotion] = emotions.get(emotion, 0) + 1
                
                # Count speakers
                speaker = unflattened.get('speaker', 'Unknown')
                speakers.add(speaker)
                
                # Count topics
                segment_topics = unflattened.get('topics', [])
                if isinstance(segment_topics, list):
                    topics.update(segment_topics)
                
                # Count types
                doc_type = unflattened.get('type', 'segment')
                types[doc_type] = types.get(doc_type, 0) + 1
            
            return {
                "total_documents": count,
                "document_types": types,
                "emotions": emotions,
                "speakers": list(speakers),
                "topics": list(topics)[:20],  # Limit to top 20
                "status": "active"
            }
            
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

def main():
    """Demo function showing how to use the RAG system"""
    
    # Initialize RAG system
    rag = FreeAudioRAG(model_name="llama3.2:3b")  # or qwen2.5:1.5b for faster responses
    
    # Check Ollama connection
    if not rag.check_ollama_connection():
        print("\n⚠️  Please install and start Ollama:")
        print("1. Install: https://ollama.ai/download")
        print("2. Start: ollama serve")
        print("3. Pull model: ollama pull llama3.2:3b")
        return
    
    # Process the audio analysis data
    analysis_file = "comprehensive_analysis_report.json"
    if os.path.exists(analysis_file):
        count = rag.process_audio_analysis(analysis_file)
        print(f"✓ Processed {count} documents")
    else:
        print(f"❌ Analysis file not found: {analysis_file}")
        return
    
    # Example queries
    example_queries = [
        {
            "query": "What segments have high confidence neutral emotions?",
            "filters": {"emotion": "neutral", "min_confidence": 0.9}
        },
        {
            "query": "What does the speaker say about Krishna consciousness?",
            "filters": {"topics": ["krishna & chanting"]}
        },
        {
            "query": "Find segments where the speaker discusses empiric philosophers",
            "filters": {}
        },
        {
            "query": "What are the symptoms of love of God mentioned in the audio?",
            "filters": {"topics": ["symptoms & freed"]}
        }
    ]
    
    print("\n" + "="*60)
    print("FREE RAG SYSTEM DEMO")
    print("="*60)
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {example['query']}")
        if example['filters']:
            print(f"Filters: {example['filters']}")
        
        result = rag.chat(example['query'], example['filters'])
        
        print(f"\nResponse: {result['response']}")
        print(f"Sources: {len(result['sources'])} segments")
        
        if result['sources']:
            print("Top sources:")
            for j, source in enumerate(result['sources'][:2], 1):
                if 'segment_id' in source:
                    print(f"  {j}. Segment {source['segment_id']} ({source['timestamp']}) - {source['emotion']}")
                else:
                    print(f"  {j}. Block {source['block_id']} summary - {source['topic']}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()
