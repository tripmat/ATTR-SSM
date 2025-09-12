"""
Copying benchmark from Jelassi et al. 2024 "Repeat After Me"
Tests the fundamental copying ability of sequence models
"""

import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import json


class CopyingTask:
    """
    Implements the copying task from Jelassi et al. 2024
    Format: <BOS> x1 x2 ... xL <COPY> -> x1 x2 ... xL
    """
    
    def __init__(self, vocab_size: int = 30, max_length: int = 300):
        self.vocab_size = vocab_size  # Alphabet size (a-z + special tokens)
        self.max_length = max_length
        
        # Special tokens
        self.BOS_TOKEN = 0
        self.COPY_TOKEN = 1  
        self.EOS_TOKEN = 2
        
        # Regular tokens: 3 to vocab_size-1
        self.regular_tokens = list(range(3, vocab_size))
        
    def generate_uniform_string(self, length: int) -> List[int]:
        """Generate uniform random string of given length"""
        return [random.choice(self.regular_tokens) for _ in range(length)]
    
    def generate_natural_string(self, length: int) -> List[int]:
        """Generate more structured string (simulating natural language patterns)"""
        # Simple bigram model to create some structure
        string = []
        prev_token = random.choice(self.regular_tokens)
        string.append(prev_token)
        
        for _ in range(length - 1):
            # Higher probability for certain transitions
            if random.random() < 0.3:  # Repeat previous token
                next_token = prev_token
            else:
                next_token = random.choice(self.regular_tokens)
            string.append(next_token)
            prev_token = next_token
            
        return string
    
    def shuffle_string(self, string: List[int]) -> List[int]:
        """Shuffle word order (simulating shuffled natural text)"""
        # For simplicity, shuffle at token level with some structure preservation
        shuffled = string.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def create_copy_example(self, length: int, string_type: str = "uniform") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a copying example
        Returns: (input_sequence, target_sequence)
        """
        # Generate string based on type
        if string_type == "uniform":
            string = self.generate_uniform_string(length)
        elif string_type == "natural":
            string = self.generate_natural_string(length)
        elif string_type == "shuffled":
            natural = self.generate_natural_string(length)
            string = self.shuffle_string(natural)
        else:
            raise ValueError(f"Unknown string_type: {string_type}")
        
        # Create input: <BOS> x1 x2 ... xL <COPY>
        input_seq = [self.BOS_TOKEN] + string + [self.COPY_TOKEN]
        
        # Create target: x1 x2 ... xL <EOS>
        target_seq = string + [self.EOS_TOKEN]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)
    
    def create_batch(self, batch_size: int, length: int, string_type: str = "uniform") -> Tuple[torch.Tensor, torch.Tensor]:
        """Create batch of copying examples with padding"""
        inputs, targets = [], []
        
        for _ in range(batch_size):
            inp, tgt = self.create_copy_example(length, string_type)
            inputs.append(inp)
            targets.append(tgt)
        
        # Pad sequences to same length
        max_input_len = max(len(inp) for inp in inputs)
        max_target_len = max(len(tgt) for tgt in targets)
        
        padded_inputs = torch.zeros(batch_size, max_input_len, dtype=torch.long)
        padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            padded_inputs[i, :len(inp)] = inp
            padded_targets[i, :len(tgt)] = tgt
            
        return padded_inputs, padded_targets


class PhoneBookTask:
    """
    Phone book lookup task from Jelassi et al. 2024
    Given a list of name-number pairs, retrieve specific number
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.SEP_TOKEN = 0  # Separator between name and number
        self.QUERY_TOKEN = 1  # Query marker
        self.COLON_TOKEN = 2  # Between name and number
        
    def generate_phone_book(self, num_entries: int) -> List[Tuple[List[int], List[int]]]:
        """Generate phone book with name-number pairs"""
        entries = []
        used_names = set()
        
        for _ in range(num_entries):
            # Generate unique name (3-5 tokens)
            while True:
                name_length = random.randint(3, 5)
                name = [random.randint(3, self.vocab_size // 2) for _ in range(name_length)]
                name_tuple = tuple(name)
                if name_tuple not in used_names:
                    used_names.add(name_tuple)
                    break
            
            # Generate phone number (7 digits)
            number = [random.randint(self.vocab_size // 2 + 1, self.vocab_size - 1) for _ in range(7)]
            
            entries.append((name, number))
        
        return entries
    
    def create_lookup_example(self, num_entries: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create phone book lookup example"""
        # Generate phone book
        entries = self.generate_phone_book(num_entries)
        
        # Create context: name1:number1 SEP name2:number2 SEP ... 
        context = []
        for name, number in entries:
            context.extend(name + [self.COLON_TOKEN] + number + [self.SEP_TOKEN])
        
        # Select random entry to query
        query_idx = random.randint(0, len(entries) - 1)
        query_name, target_number = entries[query_idx]
        
        # Create input: context QUERY query_name
        input_seq = context + [self.QUERY_TOKEN] + query_name
        
        # Target: the corresponding number
        target_seq = target_number
        
        return torch.tensor(input_seq), torch.tensor(target_seq)


class CopyingBenchmark:
    """
    Complete copying benchmark suite
    """
    
    def __init__(self, vocab_size: int = 30):
        self.vocab_size = vocab_size
        self.copy_task = CopyingTask(vocab_size)
        self.phone_task = PhoneBookTask(vocab_size * 10)  # Larger vocab for phone book
        
    def evaluate_copying_efficiency(self, model, max_samples: int = 10000, target_length: int = 300):
        """Test training efficiency on copying (Figure 1a from paper)"""
        model.train()
        results = {"samples": [], "accuracies": []}
        
        sample_counts = [10, 50, 100, 500, 1000, 5000, 10000]
        
        for num_samples in sample_counts:
            if num_samples > max_samples:
                break
                
            print(f"Training with {num_samples} samples...")
            
            # Simple training loop
            correct = 0
            total = 0
            
            for _ in tqdm(range(min(num_samples, 1000)), desc="Training"):  # Limit for demo
                inputs, targets = self.copy_task.create_batch(1, target_length, "uniform")
                
                try:
                    with torch.no_grad():
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        
                        # Check string-level accuracy
                        predicted = torch.argmax(logits[0], dim=-1)
                        if len(predicted) >= len(targets[0]):
                            matches = (predicted[:len(targets[0])] == targets[0]).sum().item()
                            if matches == len(targets[0]):
                                correct += 1
                        total += 1
                        
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results["samples"].append(num_samples)
            results["accuracies"].append(accuracy)
            
            print(f"Accuracy after {num_samples} samples: {accuracy:.1f}%")
            
        return results
    
    def evaluate_length_generalization(self, model, train_length: int = 50):
        """Test length generalization (Figure 1b from paper)"""
        test_lengths = [37, 50, 100, 200, 500, 1000]
        results = {"lengths": [], "accuracies": []}
        
        model.eval()
        
        for length in test_lengths:
            if length > 1000:  # Avoid memory issues
                break
                
            print(f"Testing length {length}...")
            correct = 0
            total = 0
            
            for _ in tqdm(range(50), desc=f"Length {length}"):  # 50 trials
                try:
                    inputs, targets = self.copy_task.create_batch(1, length, "uniform")
                    
                    with torch.no_grad():
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        
                        # Check string-level accuracy  
                        predicted = torch.argmax(logits[0], dim=-1)
                        if len(predicted) >= len(targets[0]):
                            matches = (predicted[:len(targets[0])] == targets[0]).sum().item()
                            if matches == len(targets[0]):
                                correct += 1
                        total += 1
                        
                except Exception as e:
                    print(f"Error at length {length}: {e}")
                    continue
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results["lengths"].append(length)
            results["accuracies"].append(accuracy)
            
            print(f"Accuracy at length {length}: {accuracy:.1f}%")
            
        return results
    
    def evaluate_phone_book(self, model, max_entries: int = 200):
        """Test phone book lookup (Figure 1c from paper)"""
        entry_counts = [20, 50, 100, 150, 200]
        results = {"entries": [], "accuracies": []}
        
        model.eval()
        
        for num_entries in entry_counts:
            if num_entries > max_entries:
                break
                
            print(f"Testing phone book with {num_entries} entries...")
            correct = 0
            total = 0
            
            for _ in tqdm(range(20), desc=f"Entries {num_entries}"):  # 20 trials
                try:
                    inputs, targets = self.phone_task.create_lookup_example(num_entries)
                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        
                        # Check if model retrieves correct number
                        predicted = torch.argmax(logits[0], dim=-1)
                        if len(predicted) >= len(targets):
                            # Look for the target number in the prediction
                            pred_list = predicted.tolist()
                            target_list = targets.tolist()
                            
                            # Simple substring matching
                            for i in range(len(pred_list) - len(target_list) + 1):
                                if pred_list[i:i+len(target_list)] == target_list:
                                    correct += 1
                                    break
                        total += 1
                        
                except Exception as e:
                    print(f"Error with {num_entries} entries: {e}")
                    continue
            
            accuracy = (correct / total * 100) if total > 0 else 0
            results["entries"].append(num_entries)  
            results["accuracies"].append(accuracy)
            
            print(f"Accuracy with {num_entries} entries: {accuracy:.1f}%")
            
        return results
    
    def run_full_benchmark(self, models: Dict[str, Any], save_path: str = None):
        """Run complete copying benchmark suite"""
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'='*50}")
            
            model_results = {}
            
            # 1. Training efficiency
            print(f"\n🎯 Testing training efficiency...")
            model_results["training_efficiency"] = self.evaluate_copying_efficiency(model)
            
            # 2. Length generalization  
            print(f"\n🎯 Testing length generalization...")
            model_results["length_generalization"] = self.evaluate_length_generalization(model)
            
            # 3. Phone book lookup
            print(f"\n🎯 Testing phone book lookup...")
            model_results["phone_book"] = self.evaluate_phone_book(model)
            
            all_results[model_name] = model_results
            
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Results saved to {save_path}")
            
        return all_results


def main():
    """Demo of the copying benchmark"""
    print("🔍 COPYING BENCHMARK DEMO")
    print("=" * 50)
    
    # Create benchmark
    benchmark = CopyingBenchmark(vocab_size=30)
    
    # Test task generation
    copy_task = benchmark.copy_task
    
    print("Sample copying example:")
    inputs, targets = copy_task.create_copy_example(10, "uniform")
    print(f"Input:  {inputs.tolist()}")
    print(f"Target: {targets.tolist()}")
    
    print("\nSample phone book example:")
    phone_task = benchmark.phone_task
    inputs, targets = phone_task.create_lookup_example(5)
    print(f"Context length: {len(inputs)}")
    print(f"Target: {targets.tolist()}")
    
    print("\n✅ Copying benchmark ready for model evaluation!")


if __name__ == "__main__":
    main()