# ML/DS Clean Code Quick Reference

## 🎯 Core Principles (The "Why")

1. **Code is read much more than written** - Most cost is in maintenance
2. **Leave code better than you found it** - Boy scouts rule
3. **Keep it simple** - Use half brain cells when writing vs reading
4. **Match quality to maturity** - Don't over-engineer POCs

---

## 📝 Naming Conventions (The "What")

### Variables
- ✅ **Descriptive:** `learning_rate`, `train_set`, `user_count`
- ❌ **Avoid:** `lr`, `df`, `x`, `tmp`

### Functions
- ✅ **Verb phrases:** `evaluate_model`, `load_training_data`
- ❌ **Avoid:** `load`, `get`, `process` (too generic)

### Classes
- ✅ **Noun phrases:** `Classifier`, `DataHandler`, `ModelTrainer`
- ❌ **Avoid:** `Manager`, `Processor`, `Helper`

### Constants
- ✅ **ALL_CAPS:** `MAX_EPOCHS = 100`, `LEARNING_RATE = 0.001`
- ❌ **Avoid:** Magic numbers in code

---

## 🔧 Function Design (The "How")

### Size Rules
- ✅ **Small functions:** < 20-30 lines
- ✅ **One thing only:** Single responsibility
- ✅ **Either do or answer:** No side effects

### Parameter Rules
- ✅ **Few arguments:** Avoid > 3 parameters
- ✅ **Use data structures:** Group related params

### Structure Rules
- ✅ **No side effects:** Don't modify unexpected data
- ✅ **Return early:** Avoid deep nesting
- ✅ **Don't repeat:** Extract repeated code

---

## 📁 Code Organization (The "Where")

### File Structure
```
project/
├── data_processing.py     # One concern per file
├── model_training.py       # Separate concerns
├── model_evaluation.py     # Logical grouping
└── inference.py
```

### Script Layout (Newspaper Principle)
```python
# 1. Imports (stdlib, third-party, local)
# 2. Constants (ALL_CAPS)
# 3. High-level functions (main orchestration)
# 4. Mid-level functions (specific logic)
# 5. Low-level functions (helpers)
# 6. if __name__ == "__main__": entry point
```

### Organization Principles
- **High-level at top:** Main functions first, details below
- **Related code close:** Vertical proximity matters
- **Declare close to use:** Define variables near usage

---

## 💬 Comments and Documentation (The "When")

### Comment Philosophy
- **Code should explain itself:** 90% use good names instead
- **Explain WHY not HOW:** Comments for reasoning, not mechanics
- **Keep them short:** 1-2 sentences max
- **Delete outdated:** Never leave mismatched comments
- **Don't comment out code:** Just delete, git tracks history

### When to Comment
```python
# ✅ GOOD - Explains WHY
# Using SGD because financial time series benefit from momentum
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# ✅ GOOD - Explains non-obvious logic
# Clip to [-5, 5] to prevent numerical instability
rv_clipped = np.clip(realized_volatility, -5, 5)

# ❌ BAD - Comments the obvious
i = i + 1  # increment i
```

### Docstrings (Google Style)
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-line summary of what the function does.
    
    Longer explanation if needed (optional).
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(value1, value2)
    """
```

---

## 🎨 Formatting and Style (The "Look")

### General Principles
- **Vertical openness:** Blank lines between concepts
- **Related code together:** Don't separate related logic
- **Consistent style:** Use linter/formatter
- **Less is more:** Short > long (but not at expense of clarity)

### PEP 8 Compliance
- **snake_case:** `calculate_metrics`, `train_model`
- **PascalCase:** `TimesFMModel`, `DataProcessor`
- **4 spaces:** Never tabs
- **Line length:** 79-100 characters (soft limit)
- **Imports:** Group (stdlib, third-party, local)

---

## ❌ Anti-Patterns (The "Don'ts")

### Naming Anti-Patterns
- ❌ Single letters: `x`, `y`, `data`
- ❌ Generic names: `manager`, `processor`, `helper`
- ❌ Abbreviations: `lr`, `clf`, `pred`
- ❌ Same word different meanings: `get` for fetch/compute/create

### Function Anti-Patterns
- ❌ Large functions: >50 lines
- ❌ Multiple responsibilities: Load, process, save in one function
- ❌ Many parameters: >5 parameters
- ❌ Side effects: Modify inputs unexpectedly
- ❌ Deep nesting: Hard to read

### Code Organization Anti-Patterns
- ❌ Copy-paste code: Should be functions
- ❌ Mixed concerns: Data, model, evaluation in one file
- ❌ Circular imports: Files depend on each other
- ❌ Commented code: Blocks of "backup" code

### Comment Anti-Patterns
- ❌ Commenting the obvious: `i = i + 1  # increment i`
- ❌ Outdated comments: Code changed, comments didn't
- ❌ Commented code: Blocks of "backup" code
- ❌ Noise comments: No useful information

---

## 🔬 Research-Specific Best Practices

### Experimentation Code
- ✅ Version control everything
- ✅ Track hyperparameters
- ✅ Reproducible seeds
- ✅ Save intermediate results
- ✅ Document decisions

### Data Processing
- ✅ Separate raw and processed
- ✅ Log transformations
- ✅ Validate inputs early
- ✅ No data leakage
- ✅ Document pipeline

### Model Training
- ✅ Save checkpoints
- ✅ Log all metrics
- ✅ Validate during training
- ✅ Handle failures gracefully
- ✅ Document architecture

### Code Quality Levels
1. **Total mess:** Quick POC, testing ideas (acceptable for exploration)
2. **Readable:** Others can understand (minimum for sharing)
3. **Production:** Clean, tested, documented (required for deployment)

---

## 🚀 Quick Checklist

Before committing code, ask yourself:
- [ ] Are variables/functions/classes named descriptively?
- [ ] Are functions small (< 30 lines) and do one thing?
- [ ] Is code organized logically (high-level at top)?
- [ ] Are comments explaining WHY, not HOW?
- [ ] Are there commented out code blocks to delete?
- [ ] Are constants ALL_CAPS at top of file?
- [ ] Is code formatted consistently (PEP 8)?
- [ ] Are there obvious anti-patterns to fix?
- [ ] Is code better than when you found it?

---

## 📚 Key Resources

- **Clean Code** by Robert C. Martin (Uncle Bob)
- **The Pragmatic Programmer** by Andrew Hunt and David Thomas
- **PEP 8 Style Guide** for Python Code
- **Google Python Style Guide**

---

*Last Updated: 2026-06-13*
*Universal clean code rules for ML/DS research projects*
