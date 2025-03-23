import pickle
import json
import sys

# Create a more comprehensive placeholder Vocab class
class Vocab:
    def __init__(self):
        self.itos = []  # Index to string mapping
        self.stoi = {}  # String to index mapping
        self.vectors = None
        self.unk_index = 0
        self.unk_token = "<unk>"
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)
    
    def __len__(self):
        return len(self.itos)

# Now try to load the pickle file
try:
    print("Attempting to load pickle file with dummy Vocab class...")
    with open('embeddings/vocab.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("Success! Pickle loaded.")
    print(f"Type of loaded data: {type(data)}")
    
    # Try to convert to a serializable format
    if hasattr(data, '__dict__'):
        serializable_data = data.__dict__
    else:
        serializable_data = data
    
    # Save as JSON
    with open('converted_data_option1.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)
    
    print("Data saved to converted_data_option1.json")
    
except Exception as e:
    print(f"Error in Option 1: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error traceback: {sys.exc_info()[2]}")