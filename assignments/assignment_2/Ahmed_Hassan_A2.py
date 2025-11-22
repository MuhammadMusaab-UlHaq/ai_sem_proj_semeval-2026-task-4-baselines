import json
import os
import sys
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, classification_report

class SemEvalBaseline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def load_data(self, filepath):
        """Reads JSONL file line by line."""
        data = []
        print(f"Reading data from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): 
                        data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

    def predict(self, anchor, text_a, text_b):
        """
        Returns True if text_a is closer to anchor, False otherwise.
        """
        embeddings = self.model.encode([anchor, text_a, text_b])
        anchor_emb = embeddings[0]
        a_emb = embeddings[1]
        b_emb = embeddings[2]

        score_a = util.cos_sim(anchor_emb, a_emb).item()
        score_b = util.cos_sim(anchor_emb, b_emb).item()

        # True if A is closer, False if B is closer
        prediction_bool = (score_a > score_b)
        return prediction_bool, score_a, score_b

def find_dataset(filename):
    """
    Searches for the file recursively starting from the script's directory.
    """
    # print(f"Searching for '{filename}'...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            # print(f"FOUND IT: {full_path}")
            return full_path
    
    return None

def run_baseline():
    # ---------------------------------------------------------
    # AUTO-DISCOVERY MODE
    # ---------------------------------------------------------
    target_filename = "dev_track_a.jsonl"
    input_file = find_dataset(target_filename)

    if not input_file:
        print("\n!!! CRITICAL FAILURE !!!")
        print(f"Could not find '{target_filename}' anywhere in this folder or subfolders.")
        print("1. Check if the file name is spelled exactly 'dev_track_a.jsonl'")
        print("2. Check if the 'data' folder is actually in this project directory.")
        return
    # ---------------------------------------------------------

    # Initialize Solver
    solver = SemEvalBaseline()
    dataset = solver.load_data(input_file)
    
    if not dataset:
        print("No data loaded. Exiting.")
        return

    print(f"Processing {len(dataset)} items...")
    
    predictions = []
    ground_truth = []
    
    for i, item in enumerate(dataset):
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        is_a_closer_actual = item.get('text_a_is_closer')
        
        if anchor and text_a and text_b and is_a_closer_actual is not None:
            pred_bool, score_a, score_b = solver.predict(anchor, text_a, text_b)
            predictions.append(pred_bool)
            ground_truth.append(is_a_closer_actual)
            
            if i < 3:
                print(f"\n[Item {i}]")
                print(f"Anchor: {anchor[:40]}...")
                print(f"Pred: {pred_bool} | Actual: {is_a_closer_actual}")
                print(f"Result: {'CORRECT' if pred_bool == is_a_closer_actual else 'WRONG'}")

    if ground_truth:
        acc = accuracy_score(ground_truth, predictions)
        print("\n" + "="*50)
        print(f"FINAL RESULTS")
        print("="*50)
        print(f"Total Items: {len(ground_truth)}")
        print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
        print("-" * 50)
        print(classification_report(ground_truth, predictions, target_names=['B is Closer', 'A is Closer']))

if __name__ == "__main__":
    run_baseline()