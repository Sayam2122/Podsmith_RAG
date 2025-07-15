import json
import os
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Any:
    """Load and return JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_audio_emotion_by_segment_id(audio_emotions: List[Dict], segment_id: int) -> Dict:
    """Find audio emotion data by segment ID"""
    for emotion in audio_emotions:
        if emotion.get('segment_id') == segment_id:
            return emotion
    return {}

def find_text_emotion_by_id(text_emotions: List[Dict], text_id: int) -> Dict:
    """Find text emotion data by text ID"""
    for emotion in text_emotions:
        if emotion.get('text_id') == text_id:
            return emotion
    return {}

def find_combined_emotion_by_id(combined_emotions: Dict, text_id: int) -> Dict:
    """Find combined emotion data by text ID"""
    text_emotions = combined_emotions.get('text_emotions', [])
    for emotion in text_emotions:
        if emotion.get('text_id') == text_id:
            return emotion
    return {}

def generate_comprehensive_report():
    """Generate comprehensive analysis report combining all data sources"""
    
    # Load all data files
    summaries = load_json_file('summaries.json')
    semantic_blocks = load_json_file('semantic_blocks.json')
    transcript = load_json_file('transcript.json')
    emotions_audio = load_json_file('emotions_audio.json')
    emotions_text = load_json_file('emotions_text.json')
    emotions_combined = load_json_file('emotions_combined.json')
    
    if not all([summaries, semantic_blocks, transcript, emotions_audio, emotions_text, emotions_combined]):
        print("Error: Could not load all required files")
        return
    
    # Start building the comprehensive report
    report = {
        "comprehensive_analysis": {
            "generation_timestamp": "2025-07-14",
            "overall_summary": summaries.get('global_summary', {}),
            "block_wise_analysis": []
        }
    }
    
    # Process each semantic block
    for block in semantic_blocks:
        block_id = block.get('block_id')
        print(f"Processing Block {block_id}...")
        
        # Get block summary from summaries
        block_summary = {}
        if 'block_summaries' in summaries:
            for summary in summaries['block_summaries']:
                if summary.get('block_id') == block_id:
                    block_summary = summary
                    break
        
        # Prepare block analysis structure
        block_analysis = {
            "block_id": block_id,
            "block_summary": block_summary,
            "block_content": {
                "text": block.get('text', ''),
                "start": block.get('start'),
                "end": block.get('end'),
                "duration": block.get('duration'),
                "speaker": block.get('speaker'),
                "speaker_distribution": block.get('speaker_distribution', {}),
                "segment_count": block.get('segment_count'),
                "word_count": block.get('word_count'),
                "topics": block.get('topics', []),
                "topic_probabilities": block.get('topic_probabilities', {}),
                "topic_keywords": block.get('topic_keywords', []),
                "semantic_coherence": block.get('semantic_coherence'),
                "content_density": block.get('content_density')
            },
            "original_segments": []
        }
        
        # Process each original segment in the block
        original_segments = block.get('original_segments', [])
        for segment_id in original_segments:
            # Find transcript data for this segment
            segment_data = None
            for transcript_item in transcript:
                if transcript_item.get('id') == segment_id:
                    segment_data = transcript_item
                    break
            
            if segment_data:
                # Find corresponding emotions
                audio_emotion = find_audio_emotion_by_segment_id(emotions_audio, segment_id)
                
                # For text emotions, we need to match by start/end times or text content
                text_emotion = {}
                for text_emo in emotions_text:
                    if (abs(text_emo.get('start', 0) - segment_data.get('start', 0)) < 0.1 and 
                        abs(text_emo.get('end', 0) - segment_data.get('end', 0)) < 0.1):
                        text_emotion = text_emo
                        break
                
                # Find combined emotion
                combined_emotion = {}
                if 'text_emotions' in emotions_combined:
                    for comb_emo in emotions_combined['text_emotions']:
                        if (abs(comb_emo.get('start', 0) - segment_data.get('start', 0)) < 0.1 and 
                            abs(comb_emo.get('end', 0) - segment_data.get('end', 0)) < 0.1):
                            combined_emotion = comb_emo
                            break
                
                # Compile segment information
                segment_info = {
                    "segment_id": segment_id,
                    "speaker": segment_data.get('speaker'),
                    "start": segment_data.get('start'),
                    "end": segment_data.get('end'),
                    "text": segment_data.get('text'),
                    "confidence": segment_data.get('confidence'),
                    "emotions": {
                        "audio_emotion": {
                            "emotion": audio_emotion.get('emotion', 'N/A'),
                            "confidence": audio_emotion.get('confidence', 0),
                            "all_emotions": audio_emotion.get('all_emotions', {})
                        },
                        "text_emotion": {
                            "emotion": text_emotion.get('emotion', 'N/A'),
                            "confidence": text_emotion.get('confidence', 0),
                            "all_emotions": text_emotion.get('all_emotions', {})
                        },
                        "combined_emotion": {
                            "emotion": combined_emotion.get('emotion', 'N/A'),
                            "confidence": combined_emotion.get('confidence', 0),
                            "all_emotions": combined_emotion.get('all_emotions', {})
                        }
                    }
                }
                
                block_analysis["original_segments"].append(segment_info)
        
        report["comprehensive_analysis"]["block_wise_analysis"].append(block_analysis)
    
    # Save the comprehensive report
    output_file = 'comprehensive_analysis_report.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Comprehensive analysis report saved to {output_file}")
        
        # Also create a summary version
        create_summary_report(report)
        
    except Exception as e:
        print(f"Error saving report: {e}")

def create_summary_report(full_report: Dict):
    """Create a more readable summary version of the report"""
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("COMPREHENSIVE SESSION ANALYSIS REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Overall Summary
    overall_summary = full_report["comprehensive_analysis"]["overall_summary"]
    summary_lines.append("OVERALL SUMMARY:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Executive Summary: {overall_summary.get('executive_summary', 'N/A')}")
    summary_lines.append("")
    
    if 'main_themes' in overall_summary:
        summary_lines.append("Main Themes:")
        for i, theme in enumerate(overall_summary['main_themes'][:10]):  # Show first 10 themes
            summary_lines.append(f"  {i+1}. {theme}")
        summary_lines.append("")
    
    # Block-wise Analysis
    summary_lines.append("BLOCK-WISE ANALYSIS:")
    summary_lines.append("-" * 40)
    
    for block in full_report["comprehensive_analysis"]["block_wise_analysis"]:
        block_id = block["block_id"]
        block_content = block["block_content"]
        
        summary_lines.append(f"\nBLOCK {block_id}:")
        summary_lines.append(f"  Duration: {block_content.get('duration', 0):.2f} seconds")
        summary_lines.append(f"  Speaker: {block_content.get('speaker', 'N/A')}")
        summary_lines.append(f"  Word Count: {block_content.get('word_count', 0)}")
        summary_lines.append(f"  Segment Count: {block_content.get('segment_count', 0)}")
        
        if block_content.get('topics'):
            summary_lines.append(f"  Topics: {', '.join(block_content['topics'])}")
        
        if block_content.get('topic_keywords'):
            summary_lines.append(f"  Keywords: {', '.join(block_content['topic_keywords'][:5])}")
        
        # Block summary if available
        block_summary = block.get("block_summary", {})
        if block_summary and 'summary' in block_summary:
            summary_lines.append(f"  Summary: {block_summary['summary']}")
        
        summary_lines.append(f"  Text Preview: {block_content.get('text', '')[:100]}...")
        
        # Show segment details
        summary_lines.append(f"  Segments ({len(block['original_segments'])}):")
        for i, segment in enumerate(block["original_segments"][:3]):  # Show first 3 segments
            summary_lines.append(f"    Segment {segment['segment_id']}:")
            summary_lines.append(f"      Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
            summary_lines.append(f"      Speaker: {segment['speaker']}")
            summary_lines.append(f"      Audio Emotion: {segment['emotions']['audio_emotion']['emotion']} ({segment['emotions']['audio_emotion']['confidence']:.3f})")
            summary_lines.append(f"      Text Emotion: {segment['emotions']['text_emotion']['emotion']} ({segment['emotions']['text_emotion']['confidence']:.3f})")
            summary_lines.append(f"      Text: {segment['text'][:80]}...")
        
        if len(block["original_segments"]) > 3:
            summary_lines.append(f"    ... and {len(block['original_segments']) - 3} more segments")
        
        summary_lines.append("-" * 60)
    
    # Save summary report
    try:
        with open('comprehensive_analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print("Summary report saved to comprehensive_analysis_summary.txt")
    except Exception as e:
        print(f"Error saving summary report: {e}")

if __name__ == "__main__":
    print("Starting comprehensive analysis report generation...")
    generate_comprehensive_report()
    print("Report generation completed!")
