import json
import os
from pathlib import Path

def filter_empty_results(input_file, output_file):
    """
    Filter out chunks where result_value is empty from XML processing results.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    
    try:
        # Load the original data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing file: {input_file}")
        print(f"Original chunks: {len(data['chunks'])}")
        print(f"Original found_items: {len(data['found_items'])}")
        
        # Filter chunks with non-empty result_value
        filtered_chunks = []
        for chunk in data['chunks']:
            if chunk.get('result_value') and chunk['result_value'].strip():
                filtered_chunks.append(chunk)
        
        # Filter found_items with non-empty identifier
        filtered_found_items = []
        for item in data['found_items']:
            if item.get('identifier') and item['identifier'].strip():
                filtered_found_items.append(item)
        
        # Create filtered data
        filtered_data = {
            'processing_info': {
                'total_files': data['processing_info']['total_files'],
                'total_chunks': len(filtered_chunks),
                'total_identifiers': len(filtered_found_items),
                'original_chunks': len(data['chunks']),
                'original_identifiers': len(data['found_items']),
                'filtered_out_chunks': len(data['chunks']) - len(filtered_chunks),
                'filtered_out_identifiers': len(data['found_items']) - len(filtered_found_items),
                'patterns_used': data['processing_info']['patterns_used']
            },
            'chunks': filtered_chunks,
            'found_items': filtered_found_items
        }
        
        # Save filtered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"Filtered chunks: {len(filtered_chunks)}")
        print(f"Filtered found_items: {len(filtered_found_items)}")
        print(f"Filtered out chunks: {len(data['chunks']) - len(filtered_chunks)}")
        print(f"Filtered out identifiers: {len(data['found_items']) - len(filtered_found_items)}")
        print(f"Results saved to: {output_file}")
        
        return filtered_data
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def analyze_empty_results(input_file):
    """
    Analyze what types of empty results exist in the data.
    
    Args:
        input_file (str): Path to input JSON file
    """
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nAnalyzing empty results in: {input_file}")
        print("=" * 60)
        
        # Analyze chunks with empty result_value
        empty_chunks = []
        for chunk in data['chunks']:
            if not chunk.get('result_value') or not chunk['result_value'].strip():
                empty_chunks.append(chunk)
        
        print(f"Chunks with empty result_value: {len(empty_chunks)}")
        
        if empty_chunks:
            print("\nSample empty chunks:")
            for i, chunk in enumerate(empty_chunks[:5]):  # Show first 5
                print(f"  {i+1}. Pattern: {chunk.get('pattern_name', 'N/A')}")
                print(f"     Chunk text: {chunk.get('chunk_text', 'N/A')[:100]}...")
                print(f"     Result value: '{chunk.get('result_value', 'N/A')}'")
                print()
        
        # Analyze found_items with empty identifier
        empty_found_items = []
        for item in data['found_items']:
            if not item.get('identifier') or not item['identifier'].strip():
                empty_found_items.append(item)
        
        print(f"Found items with empty identifier: {len(empty_found_items)}")
        
        if empty_found_items:
            print("\nSample empty found items:")
            for i, item in enumerate(empty_found_items[:5]):  # Show first 5
                print(f"  {i+1}. Type: {item.get('type', 'N/A')}")
                print(f"     Article ID: {item.get('article_id', 'N/A')}")
                print(f"     Identifier: '{item.get('identifier', 'N/A')}'")
                print()
        
        return empty_chunks, empty_found_items
        
    except Exception as e:
        print(f"Error analyzing {input_file}: {e}")
        return [], []

def main():
    """Main function to filter empty results from all XML processing files."""
    
    # Define input files to process
    input_files = [
        "xml_processing_results_kaggle.json",
        "xml_processing_results_5-9-ocr-credit-card.json",
        "xml_processing_results.json"
    ]
    
    all_filtered_data = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"\n{'='*60}")
            print(f"Processing: {input_file}")
            print(f"{'='*60}")
            
            # Analyze empty results first
            empty_chunks, empty_found_items = analyze_empty_results(input_file)
            
            # Filter empty results
            output_file = input_file.replace('.json', '_filtered.json')
            filtered_data = filter_empty_results(input_file, output_file)
            
            if filtered_data:
                all_filtered_data.append(filtered_data)
        else:
            print(f"File not found: {input_file}")
    
    # Create combined summary
    if all_filtered_data:
        total_original_chunks = sum(d['processing_info']['original_chunks'] for d in all_filtered_data)
        total_filtered_chunks = sum(d['processing_info']['total_chunks'] for d in all_filtered_data)
        total_original_identifiers = sum(d['processing_info']['original_identifiers'] for d in all_filtered_data)
        total_filtered_identifiers = sum(d['processing_info']['total_identifiers'] for d in all_filtered_data)
        
        combined_summary = {
            'summary': {
                'files_processed': len(all_filtered_data),
                'total_original_chunks': total_original_chunks,
                'total_filtered_chunks': total_filtered_chunks,
                'total_original_identifiers': total_original_identifiers,
                'total_filtered_identifiers': total_filtered_identifiers,
                'chunks_filtered_out': total_original_chunks - total_filtered_chunks,
                'identifiers_filtered_out': total_original_identifiers - total_filtered_identifiers,
                'filtered_files': [f.replace('.json', '_filtered.json') for f in input_files if os.path.exists(f)]
            }
        }
        
        with open("filtered_results_summary.json", 'w', encoding='utf-8') as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {len(all_filtered_data)}")
        print(f"Total original chunks: {total_original_chunks}")
        print(f"Total filtered chunks: {total_filtered_chunks}")
        print(f"Chunks filtered out: {total_original_chunks - total_filtered_chunks}")
        print(f"Total original identifiers: {total_original_identifiers}")
        print(f"Total filtered identifiers: {total_filtered_identifiers}")
        print(f"Identifiers filtered out: {total_original_identifiers - total_filtered_identifiers}")
        print(f"Combined summary saved to: filtered_results_summary.json")

if __name__ == "__main__":
    main() 