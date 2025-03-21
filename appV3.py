# appV3.py
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import os
import numpy as np
from loguru import logger

# Define Vocab class FIRST - this must match exactly what was used in training
class Vocab:
    """Simple vocabulary class for tokenization"""
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = {}

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count words
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)

        # Add top words to vocab
        for word, _ in sorted_words[:self.max_size-2]:  # -2 for PAD and UNK
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        return [self.word2idx.get(word, 1) for word in text.split()]  # 1 is <UNK>

    def save(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Define Improved ToxicityClassifier model
class ToxicityClassifier(nn.Module):
    """Improved Hybrid TextCNN + BiLSTM model for toxicity classification"""
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, max_len,
                 filter_sizes=[3, 4, 5, 6, 7], num_filters=256):
        super(ToxicityClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize with pretrained embeddings
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = True  # Fine-tune embeddings

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                     kernel_size=fs) for fs in filter_sizes
        ])

        # Batch normalization for CNN outputs
        self.bn_conv = nn.BatchNorm1d(num_filters * len(filter_sizes))

        # BiLSTM layers
        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)  # Increased dropout

        # Attention mechanism for BiLSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(2 * 256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Dense layers with improved regularization
        self.fc1 = nn.Linear(num_filters * len(filter_sizes) + 2 * 256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        # Output layers
        self.toxicity_out = nn.Linear(64, 3)  # 3 classes: not toxic, toxic, very toxic
        self.category_out = nn.Linear(64, 4)  # 4 binary categories

        # Dropout with increased rate
        self.dropout = nn.Dropout(0.5)
        self.high_dropout = nn.Dropout(0.6)  # Higher dropout for more regularization

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, max_len, embedding_dim)

        # CNN layers
        embedded_cnn = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded_cnn))  # (batch_size, num_filters, max_len - filter_size + 1)
            pool_out = F.max_pool1d(conv_out, conv_out.shape[2])  # (batch_size, num_filters, 1)
            conv_outputs.append(pool_out.squeeze(2))  # (batch_size, num_filters)

        cnn_features = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Apply batch normalization if batch size > 1
        if cnn_features.size(0) > 1:
            cnn_features = self.bn_conv(cnn_features)  # Batch normalization

        # BiLSTM layers
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch_size, max_len, 2*hidden_size)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, max_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, 2*hidden_size)

        # Get the last hidden state from both directions
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 2*hidden_size)
        
        # Add attention-weighted context to LSTM features
        lstm_features = (lstm_features + context_vector) / 2

        # Combine CNN and LSTM features
        combined = torch.cat([cnn_features, lstm_features], dim=1)

        # Dense layers with improved regularization
        x = self.dropout(combined)
        x = F.relu(self.fc1(x))
        
        # Apply batch normalization if batch size > 1
        if x.size(0) > 1:
            x = self.bn1(x)
            
        x = self.high_dropout(x)
        x = F.relu(self.fc2(x))
        
        if x.size(0) > 1:
            x = self.bn2(x)
            
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        if x.size(0) > 1:
            x = self.bn3(x)
            
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        
        if x.size(0) > 1:
            x = self.bn4(x)

        # Output layers
        toxicity_level = self.toxicity_out(x)  # (batch_size, 3)
        categories = torch.sigmoid(self.category_out(x))  # (batch_size, 4)

        return toxicity_level, categories

# Enhanced text processing and censoring utilities
class TextProcessor:
    """Enhanced text processing with censoring capabilities"""
    
    # Common profanity/toxic word lists categorized by severity
    TOXIC_WORDS = {
        'mild': ['damn', 'hell', 'crap', 'stupid', 'idiot', 'dumb', 'moron'],
        'medium': ['ass', 'asshole', 'bitch', 'shit', 'wtf', 'stfu', 'screw'],
        'severe': ['fuck', 'fucker', 'fucking', 'motherfucker', 'motherfucking', 'cock', 'dick', 'pussy', 'cunt']
    }
    
    # Words that may indicate threatening content
    THREATENING_WORDS = ['kill', 'die', 'death', 'murder', 'hurt', 'pain', 'suffer', 'hate']
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
            text = text.strip()
            return text
        return ""
    
    @staticmethod
    def contains_profanity(text, severity=None, custom_terms=None):
        """
        Check if text contains profanity of specified severity or custom terms
        
        Args:
            text (str): Text to check
            severity (str, optional): Severity level to check ('mild', 'medium', 'severe')
            custom_terms (set, optional): Set of custom profanity terms to check
            
        Returns:
            bool: True if profanity is found, False otherwise
        """
        text = text.lower()
        words = set(re.findall(r'\b\w+\b', text))
        
        # Check built-in profanity list
        if severity:
            if any(word in words for word in TextProcessor.TOXIC_WORDS.get(severity, [])):
                return True
        else:
            all_profanity = sum(TextProcessor.TOXIC_WORDS.values(), [])
            if any(word in words for word in all_profanity):
                return True
        
        # Check custom profanity terms if provided
        if custom_terms:
            # For single-word terms, check if they're in the word set
            single_word_terms = [term for term in custom_terms if ' ' not in term]
            if any(term in words for term in single_word_terms):
                return True
                
            # For multi-word terms, check if they're in the original text
            multi_word_terms = [term for term in custom_terms if ' ' in term]
            if any(term in text for term in multi_word_terms):
                return True
                
        return False
    
    @staticmethod
    def contains_threatening_content(text):
        """Check if text contains threatening words"""
        text = text.lower()
        words = set(re.findall(r'\b\w+\b', text))
        return any(word in words for word in TextProcessor.THREATENING_WORDS)
    
    @classmethod
    def load_custom_profanity_list(cls, filepath):
        """
        Load custom profanity terms from a CSV file
        
        Args:
            filepath (str): Path to CSV file containing profanity terms
            
        Returns:
            set: Set of custom profanity terms
        """
        try:
            import pandas as pd
            custom_terms = set()
            
            # Read the CSV file
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Extract all terms from the first column (regardless of column name)
            for term in df.iloc[:, 0]:
                if isinstance(term, str) and len(term.strip()) > 0:
                    custom_terms.add(term.strip().lower())
            
            logger.info(f"Loaded {len(custom_terms)} custom profanity terms from {filepath}")
            return custom_terms
        except Exception as e:
            logger.error(f"Error loading custom profanity list: {e}")
            return set()
    
    @staticmethod
    def censor_text(text, toxicity_level='not toxic', category_probs=None, custom_terms=None):
        """
        Censor text based on toxicity level, category probabilities, and custom terms
        
        Args:
            text (str): The text to censor
            toxicity_level (str): 'not toxic', 'toxic', or 'very toxic'
            category_probs (dict, optional): Dictionary of category probabilities
            custom_terms (set, optional): Set of custom profanity terms to censor
            
        Returns:
            str: Censored text
        """
        if toxicity_level == 'not toxic' and not custom_terms:
            return text  # No censoring needed if not toxic and no custom terms
        
        # Start with the original text
        censored_text = text
        
        # Compile all words to censor based on toxicity level
        words_to_censor = []
        if toxicity_level == 'toxic':
            words_to_censor.extend(TextProcessor.TOXIC_WORDS['medium'] + TextProcessor.TOXIC_WORDS['severe'])
        elif toxicity_level == 'very toxic':
            # Censor all toxic words for very toxic content
            words_to_censor.extend(sum(TextProcessor.TOXIC_WORDS.values(), []))
            words_to_censor.extend(TextProcessor.THREATENING_WORDS)
        
        # Create a regex pattern for whole-word matching of toxic words
        if words_to_censor:
            pattern = r'\b(' + '|'.join(map(re.escape, words_to_censor)) + r')\b'
            
            # Function to replace matched words with asterisks of same length
            def replace_with_asterisks(match):
                word = match.group(0)
                return '*' * len(word)
            
            # Perform the censoring
            censored_text = re.sub(pattern, replace_with_asterisks, censored_text, flags=re.IGNORECASE)
        
        # Also censor any terms from the custom profanity list
        if custom_terms:
            for term in custom_terms:
                # Skip empty terms
                if not term or len(term.strip()) == 0:
                    continue
                    
                if term.lower() in text.lower():
                    # For multi-word terms or terms with special characters, use a different approach
                    if ' ' in term or not term.isalnum():
                        # Simple replacement - case insensitive
                        pattern = re.escape(term)
                        censored_text = re.sub(pattern, '*' * len(term), censored_text, flags=re.IGNORECASE)
                    else:
                        # Word boundary for single words
                        pattern = r'\b' + re.escape(term) + r'\b'
                        censored_text = re.sub(pattern, '*' * len(term), censored_text, flags=re.IGNORECASE)
        
        return censored_text

# Utils functions for sequence preparation
def pad_sequences(sequences, max_len):
    """Pad sequences to the same length"""
    padded_seqs = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seqs.append(seq[:max_len])
        else:
            padded_seqs.append(seq + [0] * (max_len - len(seq)))
    return padded_seqs

# Pydantic models for request and response
class TextRequest(BaseModel):
    text: str
    censor_output: bool = True  # Option to request censored text

class BatchTextRequest(BaseModel):
    texts: List[str]
    censor_output: bool = True  # Option to request censored text for all texts

class ToxicityPrediction(BaseModel):
    text: str
    toxicity_level: str
    toxicity_probability: float
    category_probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]  # Added to show all class probabilities
    is_toxic: bool
    censored_text: Optional[str] = None  # Censored version of the text

class ToxicityResponse(BaseModel):
    predictions: List[ToxicityPrediction]
    model_version: str

# Create ModelService class to handle model loading and inference
class ModelService:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 200
        self.toxicity_levels = ['not toxic', 'toxic', 'very toxic']
        self.category_names = ['obscenity/profanity', 'insults', 'threatening', 'identity-based negativity']
        self.model_version = "3.0.0"  # Updated version with improved detection and censoring
        self.embedding_dim = 100
        
        # Default category thresholds (will be updated if threshold file exists)
        self.category_thresholds = [0.5, 0.5, 0.5, 0.5]
        
        # Class weights for threshold adjustment
        self.class_weights = [1.0, 1.5, 1.0]  # [not_toxic, toxic, very_toxic]
        
        # Initialize the text processor
        self.text_processor = TextProcessor()
        
        # Custom profanity terms from external list
        self.custom_profanity_terms = set()

    async def load(self, model_path, vocab_path, embedding_path, threshold_path=None, custom_profanity_path=None):
        try:
            logger.info(f"Loading model from {model_path}")
            logger.info(f"Loading vocabulary from {vocab_path}")
            logger.info(f"Loading embeddings from {embedding_path}")

            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            logger.info(f"Vocabulary loaded with {len(self.vocab.word2idx)} words")

            # Load embedding matrix
            embedding_matrix = torch.load(embedding_path, map_location=self.device)
            logger.info(f"Embedding matrix loaded with shape {embedding_matrix.shape}")

            # Initialize model
            self.model = ToxicityClassifier(
                vocab_size=len(self.vocab.word2idx),
                embedding_dim=self.embedding_dim,
                embedding_matrix=embedding_matrix,
                max_len=self.max_len,
                filter_sizes=[3, 4, 5, 6, 7],
                num_filters=256
            ).to(self.device)

            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("Model loaded successfully")
            
            # Load optimized category thresholds if available
            if threshold_path and os.path.exists(threshold_path):
                try:
                    self.category_thresholds = np.load(threshold_path).tolist()
                    logger.info(f"Loaded category thresholds: {self.category_thresholds}")
                except Exception as e:
                    logger.warning(f"Could not load thresholds from {threshold_path}: {e}")
            
            # Load custom profanity list if available
            if custom_profanity_path and os.path.exists(custom_profanity_path):
                self.custom_profanity_terms = TextProcessor.load_custom_profanity_list(custom_profanity_path)
                logger.info(f"Loaded {len(self.custom_profanity_terms)} custom profanity terms")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    async def predict(self, texts, toxicity_threshold=0.5, censor_output=True):
        """
        Enhanced prediction method with improved toxic content detection and optional censoring
        """
        if not isinstance(texts, list):
            texts = [texts]

        results = []
        for text in texts:
            # Preprocess the text
            cleaned_text = self.text_processor.clean_text(text)
            tokens = self.vocab.text_to_sequence(cleaned_text)
            padded_sequence = pad_sequences([tokens], self.max_len)
            input_tensor = torch.LongTensor(padded_sequence).to(self.device)

            # Check for explicit toxic or threatening keywords
            has_toxic_words = self.text_processor.contains_profanity(text, 'medium', self.custom_profanity_terms) or \
                             self.text_processor.contains_profanity(text, 'severe', self.custom_profanity_terms)
            has_mild_toxic_words = self.text_processor.contains_profanity(text, 'mild', self.custom_profanity_terms)
            has_threatening_words = self.text_processor.contains_threatening_content(text)
            
            # Check for custom profanity terms specifically
            has_custom_terms = len(self.custom_profanity_terms) > 0 and any(term in cleaned_text for term in self.custom_profanity_terms)

            # Get predictions
            with torch.no_grad():
                toxicity_logits, category_probs = self.model(input_tensor)
                
                # Process toxicity prediction
                toxicity_probs = F.softmax(toxicity_logits, dim=1)[0]
                
                # Store raw probabilities for all classes
                raw_probs = {
                    'not_toxic': float(toxicity_probs[0]),
                    'toxic': float(toxicity_probs[1]),
                    'very_toxic': float(toxicity_probs[2])
                }

                # IMPROVEMENT 1: Modified probability boosting
                if max(toxicity_probs[1], toxicity_probs[2]) > 0.2:
                    # Boost "Toxic" significantly more than "Very Toxic" to correct the imbalance
                    toxicity_probs[1] *= 2.0  # Boost Toxic more
                    toxicity_probs[2] *= 1.2  # Boost Very Toxic less
                    # Re-normalize
                    toxicity_probs = toxicity_probs / toxicity_probs.sum()
                
                # IMPROVEMENT 2: Keyword-based adjustments
                if has_toxic_words and toxicity_probs[0] > 0.3:
                    logger.info(f"Medium/severe toxic words detected in '{text}' - boosting toxic class probabilities")
                    toxicity_probs[1] = max(toxicity_probs[1] * 1.5, 0.4)  # Boost toxic class to at least 0.4
                    toxicity_probs = toxicity_probs / toxicity_probs.sum()  # Re-normalize
                
                if has_mild_toxic_words and toxicity_probs[0] > 0.7:
                    logger.info(f"Mild toxic words detected in '{text}' - slightly boosting toxic class probabilities")
                    toxicity_probs[1] = max(toxicity_probs[1] * 1.2, 0.2)  # Slightly boost toxic class
                    toxicity_probs = toxicity_probs / toxicity_probs.sum()  # Re-normalize
                
                if has_threatening_words and toxicity_probs[2] < 0.3:
                    logger.info(f"Threatening words detected in '{text}' - boosting very toxic probability")
                    toxicity_probs[2] = max(toxicity_probs[2] * 1.5, 0.3)  # Boost very toxic to at least 0.3
                    toxicity_probs = toxicity_probs / toxicity_probs.sum()  # Re-normalize
                
                # IMPROVEMENT 3: Custom profanity term detection
                if has_custom_terms and max(toxicity_probs[1], toxicity_probs[2]) < 0.4:
                    logger.info(f"Custom profanity term detected in '{text}' - boosting toxic probability")
                    toxicity_probs[1] = max(toxicity_probs[1] * 2.0, 0.5)  # Boost toxic to at least 0.5
                    toxicity_probs = toxicity_probs / toxicity_probs.sum()  # Re-normalize
                
                # Apply class weight adjustments
                adjusted_probs = toxicity_probs.clone()
                adjusted_probs[0] = adjusted_probs[0] * 1.0  # Don't adjust not_toxic (as requested)
                adjusted_probs[1] = adjusted_probs[1] * 1.5  # Significantly boost toxic
                adjusted_probs[2] = adjusted_probs[2] * 0.8  # Reduce very_toxic
                
                # Get initial prediction
                predicted_toxicity_idx = torch.argmax(adjusted_probs).item()
                
                # IMPROVEMENT 4: Apply decision rules for better class boundaries
                
                # If it's predicted as Very Toxic but the Toxic probability is significant
                if predicted_toxicity_idx == 2 and toxicity_probs[1] > 0.3:
                    # If the difference between Toxic and Very Toxic is small
                    if toxicity_probs[2] - toxicity_probs[1] < 0.2:
                        predicted_toxicity_idx = 1  # Change to Toxic instead
                        logger.info(f"Adjusting Very Toxic → Toxic classification for '{text}'")
                
                # Special handling for toxic vs very toxic boundary
                if toxicity_probs[1] > 0.4 and toxicity_probs[2] > 0.4:
                    # If content contains profanity but no threats, prefer Toxic over Very Toxic
                    profanity_score = float(category_probs[0][0])  # obscenity/profanity
                    threatening_score = float(category_probs[0][2])  # threatening
                    
                    if profanity_score > 0.6 and threatening_score < 0.4:
                        predicted_toxicity_idx = 1  # Classify as Toxic
                        logger.info(f"Content has profanity but low threat - classified as Toxic")
                
                # Apply threshold to toxicity - if not_toxic probability > threshold,
                # force classification to not_toxic despite what argmax says
                if toxicity_probs[0] > toxicity_threshold and predicted_toxicity_idx > 0:
                    logger.info(f"Overriding toxicity prediction for '{text}' - not_toxic prob: {toxicity_probs[0]}")
                    predicted_toxicity_idx = 0
                
                # Custom terms override - if we detect custom profanity terms, ensure it's not classified as not_toxic
                if has_custom_terms and predicted_toxicity_idx == 0:
                    logger.info(f"Custom profanity term detected in '{text}' - overriding not_toxic classification")
                    predicted_toxicity_idx = 1  # Set to Toxic at minimum
                
                # Final class selection
                predicted_toxicity = self.toxicity_levels[predicted_toxicity_idx]
                toxicity_probability = toxicity_probs[predicted_toxicity_idx].item()
                
                # Process category predictions
                category_probs = category_probs[0].cpu().numpy()
                
                # Apply optimized thresholds to category predictions
                categories_dict = {}
                for i, (name, prob, threshold) in enumerate(zip(
                    self.category_names, category_probs, self.category_thresholds
                )):
                    categories_dict[name] = float(prob)
                
                # Determine if text is toxic (level 1 or 2) or contains custom profanity terms
                is_toxic = predicted_toxicity_idx > 0 or has_custom_terms
                
                # Apply censoring if requested
                censored_text = None
                if censor_output and (is_toxic or has_custom_terms):
                    censored_text = self.text_processor.censor_text(
                        text, 
                        toxicity_level=predicted_toxicity,
                        category_probs=categories_dict,
                        custom_terms=self.custom_profanity_terms
                    )

            results.append(
                ToxicityPrediction(
                    text=text,
                    toxicity_level=predicted_toxicity,
                    toxicity_probability=toxicity_probability,
                    category_probabilities=categories_dict,
                    raw_probabilities=raw_probs,
                    is_toxic=is_toxic,
                    censored_text=censored_text
                )
            )
        
        return results

# Initialize model service
model_service = ModelService()

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for initializing and cleaning up resources"""
    # Startup: Load model and resources
    # Set model paths (modify these to match your deployment)
    model_path = os.getenv("MODEL_PATH", "models/final_model.pt")#v2/best_model.pt
    vocab_path = os.getenv("VOCAB_PATH", "embeddings/vocab.pkl")#v2/vocab.pkl
    embedding_path = os.getenv("EMBEDDING_PATH", "embeddings/embedding_matrix.pt")#v2/embedding_matrix.pt
    threshold_path = os.getenv("THRESHOLD_PATH", "embeddings/optimal_category_thresholds.npy")#v2/optimal_category_thresholds.npy
    custom_profanity_path = os.getenv("CUSTOM_PROFANITY_PATH", "embeddings/merge-profanity.csv")#v2/merge-profanity.csv
    
    logger.info(f"Starting up with model path: {model_path}")
    logger.info(f"Vocabulary path: {vocab_path}")
    logger.info(f"Embedding path: {embedding_path}")
    logger.info(f"Threshold path: {threshold_path}")
    logger.info(f"Custom profanity path: {custom_profanity_path}")
    
    # Check for filename with space
    if ' ' in vocab_path:
        alt_path = vocab_path.replace(' ', '')
        if os.path.exists(alt_path):
            logger.info(f"Found vocabulary at {alt_path} (without space)")
            vocab_path = alt_path
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
    else:    
        if not os.path.exists(vocab_path):
            logger.error(f"Vocabulary file not found at {vocab_path}")
            
            # Try to find any vocab pickle file
            dir_path = os.path.dirname(vocab_path) or '.'
            for file in os.listdir(dir_path):
                if file.startswith('vocab') and file.endswith('.pkl'):
                    alt_path = os.path.join(dir_path, file)
                    logger.info(f"Found alternative vocabulary file: {alt_path}")
                    vocab_path = alt_path
                    break
            
        if not os.path.exists(embedding_path):
            logger.error(f"Embedding file not found at {embedding_path}")
        else:
            # Check if custom profanity file exists
            if not os.path.exists(custom_profanity_path):
                logger.warning(f"Custom profanity file not found at {custom_profanity_path}")
                custom_profanity_path = None
            
            # Load model
            success = await model_service.load(
                model_path, 
                vocab_path, 
                embedding_path, 
                threshold_path, 
                custom_profanity_path
            )
            if not success:
                logger.error("Failed to load model. API will return errors on prediction endpoints.")
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources
    logger.info("Shutting down the application")
    # Add any cleanup code here if needed

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Toxicity Classification API",
    description="API for classifying text toxicity with improved detection and content censoring",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Toxicity Classification API with Improved Detection and Censoring",
        "model_version": model_service.model_version,
        "status": "model loaded" if model_service.model is not None else "model not loaded",
        "custom_profanity": f"{len(model_service.custom_profanity_terms)} terms loaded" if model_service.custom_profanity_terms else "not loaded",
        "endpoints": {
            "/predict": "POST endpoint for classifying a single text",
            "/batch-predict": "POST endpoint for classifying multiple texts",
            "/health": "GET endpoint for API health check",
            "/test": "GET endpoint for quick model testing",
            "/censor": "POST endpoint for censoring text without showing full analytics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_service.model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {
        "status": "ok", 
        "message": "API is healthy and model is loaded",
        "custom_profanity_loaded": len(model_service.custom_profanity_terms) > 0
    }

@app.post("/predict", response_model=ToxicityResponse)
async def predict_toxicity(
    request: TextRequest, 
    toxicity_threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Predict toxicity level and categories for input text
    
    Returns toxicity level (not toxic, toxic, very toxic) and probabilities
    for four categories: obscenity/profanity, insults, threatening, and identity-based negativity.
    Now includes censored text output.
    
    - **toxicity_threshold**: Optional threshold (0.0-1.0) to adjust sensitivity for toxicity detection
    - Request can include **censor_output** flag to control text censoring
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = await model_service.predict(
        request.text, 
        toxicity_threshold=toxicity_threshold,
        censor_output=request.censor_output
    )
    
    return ToxicityResponse(
        predictions=predictions,
        model_version=model_service.model_version
    )

@app.post("/batch-predict", response_model=ToxicityResponse)
async def batch_predict_toxicity(
    request: BatchTextRequest, 
    toxicity_threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Batch prediction endpoint for multiple texts
    
    Returns toxicity levels, category probabilities, and optionally censored texts
    for multiple input texts
    
    - **toxicity_threshold**: Optional threshold (0.0-1.0) to adjust sensitivity
    - Request can include **censor_output** flag to control text censoring
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100 texts")
    
    predictions = await model_service.predict(
        request.texts, 
        toxicity_threshold=toxicity_threshold,
        censor_output=request.censor_output
    )
    
    return ToxicityResponse(
        predictions=predictions,
        model_version=model_service.model_version
    )

class CensorRequest(BaseModel):
    text: str
    level: str = "auto"  # "auto", "light", "medium", "heavy"

class CensorResponse(BaseModel):
    original_text: str
    censored_text: str
    toxicity_level: str
    is_toxic: bool

@app.post("/censor", response_model=CensorResponse)
async def censor_text(request: CensorRequest):
    """
    Censor toxic content in text
    
    Simplified endpoint that focuses on text censoring without returning detailed analytics
    
    - **level**: Censoring level (auto, light, medium, heavy)
      - auto: censoring based on detected toxicity level
      - light: censors only severe profanity
      - medium: censors medium and severe profanity
      - heavy: censors all profanity including mild terms
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check if there are any custom profanity terms
    has_custom_terms = any(term in request.text.lower() for term in model_service.custom_profanity_terms)
    
    # Get toxicity prediction first
    predictions = await model_service.predict(request.text, censor_output=False)
    prediction = predictions[0]
    
    # Determine censoring level
    if request.level != "auto":
        # Manual censoring based on requested level
        censored_text = request.text
        
        if request.level == "light":
            # Censor only severe profanity
            for word in model_service.text_processor.TOXIC_WORDS['severe']:
                pattern = r'\b' + re.escape(word) + r'\b'
                censored_text = re.sub(pattern, '*' * len(word), censored_text, flags=re.IGNORECASE)
            
            # Always censor custom terms
            if model_service.custom_profanity_terms:
                censored_text = model_service.text_processor.censor_text(
                    censored_text, 
                    toxicity_level='not toxic',  # No additional censoring
                    custom_terms=model_service.custom_profanity_terms
                )
        
        elif request.level == "medium":
            # Censor medium and severe profanity
            for severity in ['medium', 'severe']:
                for word in model_service.text_processor.TOXIC_WORDS[severity]:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    censored_text = re.sub(pattern, '*' * len(word), censored_text, flags=re.IGNORECASE)
            
            # Always censor custom terms
            if model_service.custom_profanity_terms:
                censored_text = model_service.text_processor.censor_text(
                    censored_text, 
                    toxicity_level='not toxic',  # No additional censoring
                    custom_terms=model_service.custom_profanity_terms
                )
        
        elif request.level == "heavy":
            # Censor all profanity including mild terms
            for severity in ['mild', 'medium', 'severe']:
                for word in model_service.text_processor.TOXIC_WORDS[severity]:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    censored_text = re.sub(pattern, '*' * len(word), censored_text, flags=re.IGNORECASE)
            
            # Also censor threatening words in heavy mode
            for word in model_service.text_processor.THREATENING_WORDS:
                pattern = r'\b' + re.escape(word) + r'\b'
                censored_text = re.sub(pattern, '*' * len(word), censored_text, flags=re.IGNORECASE)
            
            # Always censor custom terms
            if model_service.custom_profanity_terms:
                censored_text = model_service.text_processor.censor_text(
                    censored_text, 
                    toxicity_level='not toxic',  # No additional censoring
                    custom_terms=model_service.custom_profanity_terms
                )
    else:
        # Auto censoring based on detected toxicity level
        censored_text = model_service.text_processor.censor_text(
            request.text,
            toxicity_level=prediction.toxicity_level,
            category_probs=prediction.category_probabilities,
            custom_terms=model_service.custom_profanity_terms
        )
    
    # Update is_toxic if we have custom terms
    is_toxic = prediction.is_toxic or has_custom_terms
    
    return CensorResponse(
        original_text=request.text,
        censored_text=censored_text if is_toxic else request.text,
        toxicity_level=prediction.toxicity_level,
        is_toxic=is_toxic
    )

@app.get("/test", response_model=ToxicityResponse)
async def test_model(
    toxicity_threshold: float = Query(0.5, ge=0.0, le=1.0),
    censor: bool = Query(True, description="Whether to censor toxic content in output")
):
    """
    Test endpoint to quickly verify model performance with predefined examples
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Sample texts across the toxicity spectrum
    test_texts = [
        "Hello, nice to meet you!",
        "Have a great day",
        "I disagree with your opinion",
        "This product is terrible, I hate it",
        "You are an idiot and nobody likes you",
        "fuck off and leave me alone",
        "bitch what the fuck is wrong with you",
        "I will find you and hurt you badly"
    ]
    
    predictions = await model_service.predict(
        test_texts, 
        toxicity_threshold=toxicity_threshold,
        censor_output=censor
    )
    
    return ToxicityResponse(
        predictions=predictions,
        model_version=model_service.model_version
    )

# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("appV3:app", host="127.0.0.1", port=8000, reload=True)
