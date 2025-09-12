"""
In-Context Learning benchmark tasks for ATTR-SSM evaluation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


@dataclass
class ICLExample:
    """Single in-context learning example"""
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: Dict[str, Any] = None


class ICLTask(ABC):
    """Base class for in-context learning tasks"""
    
    def __init__(self, name: str, seq_len: int = 128):
        self.name = name
        self.seq_len = seq_len
    
    @abstractmethod
    def generate_examples(
        self, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        """Generate support and query examples"""
        pass
    
    @abstractmethod
    def evaluate_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model predictions"""
        pass


class LinearRegressionTask(ICLTask):
    """
    Few-shot linear regression: y = ax + b
    Model needs to learn the relationship from support examples
    """
    
    def __init__(
        self, 
        x_range: Tuple[float, float] = (-2.0, 2.0),
        noise_std: float = 0.1,
        **kwargs
    ):
        super().__init__("linear_regression", **kwargs)
        self.x_range = x_range
        self.noise_std = noise_std
    
    def generate_examples(
        self, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        
        support_examples = []
        query_examples = []
        
        for _ in range(batch_size):
            # Sample random linear function parameters
            a = np.random.uniform(-2.0, 2.0)
            b = np.random.uniform(-1.0, 1.0)
            
            # Generate support set
            support_x = np.random.uniform(*self.x_range, n_support)
            support_y = a * support_x + b + np.random.normal(0, self.noise_std, n_support)
            
            # Generate query set  
            query_x = np.random.uniform(*self.x_range, n_query)
            query_y = a * query_x + b + np.random.normal(0, self.noise_std, n_query)
            
            # Create formatted sequences for language modeling
            # Format: "x=1.5 y=3.2 x=0.8 y=2.1 ... x=? y="
            support_seq = []
            for xi, yi in zip(support_x, support_y):
                support_seq.extend([f"x={xi:.2f}", f"y={yi:.2f}"])
            
            query_seq = []
            for xi in query_x:
                query_seq.extend([f"x={xi:.2f}", "y="])
            
            # Convert to token indices (simplified - using hash for now)
            def tokenize(seq):
                vocab = {}
                tokens = []
                for token in seq:
                    if token not in vocab:
                        vocab[token] = len(vocab)
                    tokens.append(vocab[token])
                return torch.tensor(tokens), vocab
            
            # For simplicity, we'll use continuous values directly
            support_inputs = torch.tensor(support_x, dtype=torch.float32)
            support_targets = torch.tensor(support_y, dtype=torch.float32)
            query_inputs = torch.tensor(query_x, dtype=torch.float32)
            query_targets = torch.tensor(query_y, dtype=torch.float32)
            
            support_examples.append(ICLExample(
                inputs=support_inputs,
                targets=support_targets,
                metadata={"a": a, "b": b, "task": "linear_regression"}
            ))
            
            query_examples.append(ICLExample(
                inputs=query_inputs,
                targets=query_targets,
                metadata={"a": a, "b": b, "task": "linear_regression"}
            ))
        
        return support_examples, query_examples
    
    def evaluate_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        return {"mse": mse, "mae": mae}


class PatternCopyingTask(ICLTask):
    """
    Pattern copying task: learn to copy specific patterns
    Given examples of pattern transformations, apply to new patterns
    """
    
    def __init__(
        self, 
        vocab_size: int = 50,
        pattern_length: int = 8,
        **kwargs
    ):
        super().__init__("pattern_copying", **kwargs)
        self.vocab_size = vocab_size
        self.pattern_length = pattern_length
    
    def generate_examples(
        self, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        
        support_examples = []
        query_examples = []
        
        for _ in range(batch_size):
            # Choose transformation rule (e.g., shift by constant, reverse, etc.)
            transform_type = random.choice(["shift", "reverse", "double"])
            shift_amount = random.randint(1, 5) if transform_type == "shift" else 0
            
            # Generate support examples
            support_inputs = []
            support_targets = []
            
            for _ in range(n_support):
                # Generate random pattern
                pattern = torch.randint(0, self.vocab_size, (self.pattern_length,))
                
                # Apply transformation
                if transform_type == "shift":
                    transformed = (pattern + shift_amount) % self.vocab_size
                elif transform_type == "reverse":
                    transformed = torch.flip(pattern, [0])
                elif transform_type == "double":
                    transformed = torch.cat([pattern, pattern])
                
                support_inputs.append(pattern)
                support_targets.append(transformed)
            
            # Generate query examples
            query_inputs = []
            query_targets = []
            
            for _ in range(n_query):
                pattern = torch.randint(0, self.vocab_size, (self.pattern_length,))
                
                if transform_type == "shift":
                    transformed = (pattern + shift_amount) % self.vocab_size
                elif transform_type == "reverse":
                    transformed = torch.flip(pattern, [0])
                elif transform_type == "double":
                    transformed = torch.cat([pattern, pattern])
                
                query_inputs.append(pattern)
                query_targets.append(transformed)
            
            support_examples.append(ICLExample(
                inputs=torch.stack(support_inputs),
                targets=torch.stack(support_targets),
                metadata={"transform": transform_type, "shift": shift_amount}
            ))
            
            query_examples.append(ICLExample(
                inputs=torch.stack(query_inputs),
                targets=torch.stack(query_targets),
                metadata={"transform": transform_type, "shift": shift_amount}
            ))
        
        return support_examples, query_examples
    
    def evaluate_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        # Exact match accuracy
        if predictions.shape != targets.shape:
            accuracy = 0.0
        else:
            correct = torch.all(predictions == targets, dim=-1)
            accuracy = torch.mean(correct.float()).item()
        
        return {"accuracy": accuracy}


class InductionHeadsTask(ICLTask):
    """
    Induction heads task: [A][B]...[A] -> should predict [B]
    Tests associative recall ability
    """
    
    def __init__(
        self, 
        vocab_size: int = 100,
        seq_len: int = 64,
        **kwargs
    ):
        super().__init__("induction_heads", seq_len=seq_len, **kwargs)
        self.vocab_size = vocab_size
    
    def generate_examples(
        self, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        
        support_examples = []
        query_examples = []
        
        for _ in range(batch_size):
            # Create sequence with repeated patterns
            sequence = []
            pattern_dict = {}  # A -> B mapping
            
            # Generate support patterns
            for _ in range(n_support):
                A = random.randint(1, self.vocab_size - 1)  # Avoid 0 (special token)
                B = random.randint(1, self.vocab_size - 1)
                
                while A == B or A in pattern_dict:
                    A = random.randint(1, self.vocab_size - 1)
                    B = random.randint(1, self.vocab_size - 1)
                
                pattern_dict[A] = B
                sequence.extend([A, B])
            
            # Add some random tokens
            for _ in range(10):
                sequence.append(random.randint(1, self.vocab_size - 1))
            
            # Generate query: add A, expect B
            query_pairs = []
            for A, B in list(pattern_dict.items())[:n_query]:
                sequence.append(A)
                query_pairs.append((len(sequence) - 1, B))  # (position, expected_token)
            
            # Pad sequence to fixed length
            if len(sequence) < self.seq_len:
                sequence.extend([0] * (self.seq_len - len(sequence)))
            else:
                sequence = sequence[:self.seq_len]
            
            inputs = torch.tensor(sequence[:-1])  # Input sequence
            targets = torch.tensor(sequence[1:])   # Target sequence (shifted)
            
            support_examples.append(ICLExample(
                inputs=inputs,
                targets=targets,
                metadata={"patterns": pattern_dict, "query_pairs": query_pairs}
            ))
            
            # For this task, query examples are the same as support
            query_examples.append(ICLExample(
                inputs=inputs,
                targets=targets,
                metadata={"patterns": pattern_dict, "query_pairs": query_pairs}
            ))
        
        return support_examples, query_examples
    
    def evaluate_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        # Overall next-token prediction accuracy
        correct = (predictions.argmax(dim=-1) == targets)
        accuracy = correct.float().mean().item()
        
        return {"accuracy": accuracy}


class SparseParityTask(ICLTask):
    """
    Sparse parity learning: predict XOR of specific positions
    Tests ability to learn positional logic
    """
    
    def __init__(
        self, 
        seq_len: int = 32,
        n_positions: int = 4,
        **kwargs
    ):
        super().__init__("sparse_parity", seq_len=seq_len, **kwargs)
        self.n_positions = n_positions
    
    def generate_examples(
        self, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        
        support_examples = []
        query_examples = []
        
        for _ in range(batch_size):
            # Choose random positions for parity computation
            parity_positions = random.sample(range(self.seq_len - 1), self.n_positions)
            
            # Generate support examples
            support_inputs = []
            support_targets = []
            
            for _ in range(n_support):
                # Generate binary sequence
                seq = torch.randint(0, 2, (self.seq_len,))
                
                # Compute parity
                parity = sum(seq[pos].item() for pos in parity_positions) % 2
                seq[-1] = parity  # Last position is the parity
                
                support_inputs.append(seq[:-1])
                support_targets.append(torch.tensor([parity]))
            
            # Generate query examples
            query_inputs = []
            query_targets = []
            
            for _ in range(n_query):
                seq = torch.randint(0, 2, (self.seq_len,))
                parity = sum(seq[pos].item() for pos in parity_positions) % 2
                
                query_inputs.append(seq[:-1])
                query_targets.append(torch.tensor([parity]))
            
            support_examples.append(ICLExample(
                inputs=torch.stack(support_inputs),
                targets=torch.stack(support_targets),
                metadata={"parity_positions": parity_positions}
            ))
            
            query_examples.append(ICLExample(
                inputs=torch.stack(query_inputs),
                targets=torch.stack(query_targets),
                metadata={"parity_positions": parity_positions}
            ))
        
        return support_examples, query_examples
    
    def evaluate_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        pred_classes = (predictions > 0.5).long()
        accuracy = (pred_classes == targets).float().mean().item()
        return {"accuracy": accuracy}


class ICLBenchmark:
    """
    Complete benchmark suite for in-context learning evaluation
    """
    
    def __init__(self):
        self.tasks = {
            "linear_regression": LinearRegressionTask(),
            "pattern_copying": PatternCopyingTask(),
            "induction_heads": InductionHeadsTask(),
            "sparse_parity": SparseParityTask(),
        }
    
    def run_task(
        self, 
        task_name: str, 
        n_support: int, 
        n_query: int, 
        batch_size: int = 1
    ) -> Tuple[List[ICLExample], List[ICLExample]]:
        """Run a single task"""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        task = self.tasks[task_name]
        return task.generate_examples(n_support, n_query, batch_size)
    
    def evaluate_task(
        self, 
        task_name: str, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate predictions for a task"""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        task = self.tasks[task_name]
        return task.evaluate_predictions(predictions, targets)


def visualize_linear_regression_task():
    """Visualize the linear regression task"""
    task = LinearRegressionTask()
    support, query = task.generate_examples(n_support=5, n_query=10)
    
    support_ex = support[0]
    query_ex = query[0]
    
    plt.figure(figsize=(10, 6))
    
    # Plot support examples
    plt.scatter(support_ex.inputs, support_ex.targets, 
               color='blue', s=100, label='Support', marker='o')
    
    # Plot query examples
    plt.scatter(query_ex.inputs, query_ex.targets, 
               color='red', s=100, label='Query', marker='x')
    
    # Plot true function
    x_line = np.linspace(-2, 2, 100)
    a, b = support_ex.metadata['a'], support_ex.metadata['b']
    y_line = a * x_line + b
    plt.plot(x_line, y_line, 'g--', alpha=0.7, label=f'True: y={a:.2f}x+{b:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('Few-shot Linear Regression Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_task.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test all tasks
    benchmark = ICLBenchmark()
    
    print("Testing ICL Benchmark Tasks")
    print("=" * 50)
    
    for task_name in benchmark.tasks.keys():
        print(f"\nTesting {task_name}...")
        try:
            support, query = benchmark.run_task(task_name, n_support=3, n_query=2)
            print(f"✓ {task_name}: Generated {len(support)} support, {len(query)} query examples")
            
            # Print example shapes
            s_ex = support[0]
            q_ex = query[0] 
            print(f"  Support input shape: {s_ex.inputs.shape}")
            print(f"  Support target shape: {s_ex.targets.shape}")
            print(f"  Metadata: {list(s_ex.metadata.keys()) if s_ex.metadata else 'None'}")
            
        except Exception as e:
            print(f"✗ {task_name}: Error - {e}")
    
    print(f"\n✓ All tasks tested successfully!")
    
    # Create visualization
    try:
        visualize_linear_regression_task()
        print("✓ Created linear regression visualization")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")