# Word2Vec from Scratch

A clean, educational implementation of Word2Vec using only NumPy. This project demonstrates how word embeddings work by implementing the famous Word2Vec algorithm from the ground up.

## What is Word2Vec?

Word2Vec is a technique that turns words into numbers (vectors) in a way that captures their meaning. Words with similar meanings end up close to each other in vector space. This lets us do cool things like:

- **Find similar words**: "king" is similar to "queen", "monarch"
- **Word arithmetic**: king - man + woman ≈ queen
- **Understand context**: Words that appear in similar contexts get similar representations

## Features

- **Pure NumPy implementation** - No deep learning frameworks needed
- **Two training methods**:
  - Full softmax (pedagogical, follows the math directly)
  - Negative sampling (efficient, practical for large vocabularies)
- **Built-in evaluation** - Test on Google's word analogy dataset
- **Loss visualization** - Track and plot training progress
- **Clean code** - Well-documented and easy to understand

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ahmedehabb/Word2vec.git
cd Word2vec

# Install dependencies
pip install numpy datasets matplotlib
```

### Basic Usage

**Train a simple model:**
```bash
python word2vec_numpy.py --mode text8 --dim 100 --epochs 3 --loss neg --plot_loss
```

**Train with evaluation:**
```bash
python word2vec_numpy.py \
  --mode text8 \
  --max_tokens 500000 \
  --vocab_size 10000 \
  --dim 100 \
  --epochs 3 \
  --loss neg \
  --plot_loss \
  --eval_analogies
```

### Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | `toy`, `text8` | `toy` | Dataset to use |
| `--max_tokens` | integer | `200000` | Number of tokens to use from text8 |
| `--vocab_size` | integer | `5000` | Maximum vocabulary size |
| `--dim` | integer | `100` | Embedding dimension |
| `--window` | integer | `2` | Context window size |
| `--epochs` | integer | `3` | Number of training epochs |
| `--lr` | float | `0.025` | Learning rate |
| `--loss` | `softmax`, `neg` | `softmax` | Loss function type |
| `--neg_k` | integer | `5` | Number of negative samples (for neg loss) |
| `--eval_analogies` | flag | - | Evaluate on word analogies |
| `--plot_loss` | flag | - | Plot and save loss curves |

## How It Works

### The Math (Simplified)

Word2Vec learns word embeddings by predicting context words. For each word pair (center, context):

1. **Forward pass**: Calculate probability that context word appears near center word
   ```
   P(context|center) = exp(u_context · v_center) / Σ exp(u_w · v_center)
   ```

2. **Loss**: Negative log probability (we want to maximize probability)
   ```
   Loss = -log P(context|center)
   ```

3. **Backward pass**: Update vectors to increase probability of actual context words

### Two Training Methods

**Full Softmax** - Exact but slow:
- Computes probability over entire vocabulary
- Best for small vocabularies (<10k words)
- Pedagogically clear

**Negative Sampling** - Fast and practical:
- Only updates a few negative examples per positive
- Scales to large vocabularies (100k+ words)
- Recommended for real applications

## Example Output

After training, you'll see:

```
epoch 0...
  progress: 5000/50000 (10.0%), avg_loss=3.2145, last_loss=3.1987
  progress: 10000/50000 (20.0%), avg_loss=2.8734, last_loss=2.8456
  ...
epoch=0 loss=2.4523 pairs=50000 V=5000 dim=100

analogies accuracy: semantic: 15.23%, syntactic: 32.45%; evaluated=850 skipped=150
```

If you used `--plot_loss`, you'll get a nice visualization showing how loss decreases during training!

## Understanding the Code

### Key Components

```python
# 1. Model classes
SoftmaxModel         # Full softmax implementation
NegSamplingModel     # Negative sampling implementation

# 2. Core functions
sgd_step()           # One epoch of training
init_params()        # Initialize word vectors
generate_pairs()     # Create (center, context) training pairs

# 3. Evaluation
analogy_accuracy()   # Test on word analogies
```

### Architecture

The code maintains two sets of vectors for each word:
- **Center vectors** (`v_c`) - Used when word is the center/input
- **Context vectors** (`u_o`) - Used when word is in the context/output

This separation is standard in Word2Vec and helps with optimization.

## Evaluation

Use the `--eval_analogies` flag to test your embeddings on word analogy tasks:

**Semantic analogies:**
- king : queen :: man : woman
- Paris : France :: London : England

**Syntactic analogies:**
- good : better :: bad : worse
- walk : walking :: swim : swimming

The model solves these by vector arithmetic: `king - man + woman ≈ queen`

## Tips for Best Results

1. **Start small** - Test with `--max_tokens 100000` first
2. **Use negative sampling** - Much faster than softmax for large vocabularies
3. **Increase dimensions** - 100-300 dimensions work well for most tasks
4. **More data helps** - More tokens = better embeddings (up to a point)
5. **Check the loss plot** - Should decrease smoothly

## Example Experiments

**Quick test (1-2 minutes):**
```bash
python word2vec_numpy.py --mode text8 --max_tokens 100000 --dim 50 --loss neg --plot_loss
```

**Medium training (~10 minutes):**
```bash
python word2vec_numpy.py --mode text8 --max_tokens 1000000 --vocab_size 20000 --dim 100 --epochs 5 --loss neg --plot_loss --eval_analogies
```

**Large training (~1 hour):**
```bash
python word2vec_numpy.py --mode text8 --max_tokens 10000000 --vocab_size 50000 --dim 300 --epochs 3 --loss neg --eval_analogies
```

## Project Structure

```
.
├── word2vec_numpy.py    # Main implementation
├── validations.py        # Analogy evaluation code
├── data/                 # Downloaded datasets
│   └── questions-words.txt
└── README.md
```

## Technical Details

**Optimizations:**
- Numerically stable softmax (shifted exponentials)
- Efficient negative sampling
- Vectorized operations with NumPy

**Dataset:**
- Uses text8 (cleaned Wikipedia text)
- Automatically downloaded via Hugging Face datasets
- 100M characters of clean English text

## Learning Resources

Want to understand Word2Vec better?

- **Original Paper**: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- **Tutorial**: Word2Vec Tutorial - The Skip-Gram Model (Chris McCormick)
- **Visualization**: Play with word embeddings at [Embedding Projector](https://projector.tensorflow.org/)

## Common Issues & Solutions

**Out of memory?**
- Reduce `--vocab_size`
- Reduce `--max_tokens`
- Use negative sampling instead of softmax

**Loss not decreasing?**
- Check learning rate (try 0.01-0.05)
- Ensure you have enough data
- Try more epochs

**Low accuracy on analogies?**
- Increase `--dim` (try 200-300)
- Use more training data
- Train for more epochs

## Contributing

This is an educational project! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation
