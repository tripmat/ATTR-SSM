"""
Evaluation utilities for the academic replication.
Clean evaluation functions following ML research standards.
"""

import torch
import numpy as np
from tqdm import tqdm


def evaluate_model_simple(model, task, test_lengths, n_samples=20):
    """
    Simplified evaluation for clean results.
    
    Args:
        model: Trained PyTorch model
        task: CopyingTask instance
        test_lengths: List of sequence lengths to evaluate
        n_samples: Number of samples per length
        
    Returns:
        dict: Results mapping length -> accuracy
    """
    model.eval()
    # Derive the model device to send evaluation inputs appropriately
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    results = {}
    
    for length in tqdm(test_lengths, desc="Evaluating lengths"):
        correct = 0
        total = n_samples
        
        with torch.no_grad():
            for _ in range(n_samples):
                try:
                    input_seq, target_seq = task.create_copy_example(length, "uniform")
                    
                    # Handle tensor/list conversion
                    if isinstance(input_seq, torch.Tensor):
                        input_list = input_seq.tolist()
                        target_list = target_seq.tolist()
                    else:
                        input_list = input_seq
                        target_list = target_seq
                    
                    # Use teacher forcing: create full sequence for evaluation
                    full_sequence = input_list + target_list
                    full_tensor = torch.tensor(full_sequence, device=device).unsqueeze(0)
                    
                    outputs = model(full_tensor)
                    logits = outputs["logits"]
                    
                    # Use same indexing as training: target starts at len(input_list)
                    target_start = len(input_list)
                    if target_start + len(target_list) <= logits.size(1):
                        # Match training: shift by -1 for next token prediction
                        pred_logits = logits[0, target_start-1:target_start-1+len(target_list)]
                        # Ensure CPU numpy conversion works on GPU devices
                        predictions = pred_logits.argmax(dim=-1).detach().cpu().numpy()
                        target_np = np.array(target_list)
                        
                        if np.array_equal(predictions, target_np):
                            correct += 1
                except:
                    continue
        
        accuracy = correct / total
        results[length] = accuracy
        
    return results
