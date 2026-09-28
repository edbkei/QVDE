# Model Performance Comparison Guide

## 🎯 Quick Reference: How to Compare Models

### TL;DR - Fast Commands

```bash
# View all models with metrics
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# Check which model is currently loaded
docker logs custom-lstm-detector | grep "CURRENT"

# View best model selection
docker logs custom-lstm-detector | grep "BEST MODEL"
```

---

## 📊 Understanding Model Performance Metrics

### Key Metrics Explained

| Metric | Description | What It Means | Target Value |
|--------|-------------|---------------|--------------|
| **Accuracy** | Percentage of correct predictions | Overall correctness | **>90%** |
| **Precision** | Of all predicted falls, how many were real? | False alarm rate | **>85%** |
| **Recall** | Of all real falls, how many were detected? | Miss rate | **>90%** |
| **F1 Score** | Harmonic mean of precision & recall | Overall performance | **>0.90** |

### Why Each Metric Matters

**Accuracy**: Overall correctness
- 95% accuracy = 95 out of 100 predictions correct
- High accuracy = good overall performance

**Precision**: Avoiding false alarms
- 90% precision = 9 out of 10 fall alerts are real
- Low precision = too many false alarms (alert fatigue)

**Recall**: Catching all real falls
- 95% recall = detect 95 out of 100 real falls
- Low recall = missing falls (DANGEROUS!)
- **Most important metric for fall detection!**

**F1 Score**: Balance of precision and recall
- Single number summarizing performance
- Good for comparing models
- Higher is always better

---

## 🏆 Best Model vs Latest Model

### Best Model (Recommended for Production)
- **Selected by**: Highest validation accuracy
- **Proven**: Best tested performance
- **Stable**: Validated on test data
- ✅ **Use for**: Production operations
- ✅ **Use for**: Critical detection
- ✅ **Use for**: Deployed systems

### Latest Model (Use for Testing)
- **Selected by**: Most recent training date
- **Unproven**: May or may not be better
- **Experimental**: Needs validation
- ⚠️ **Use for**: Testing new training data
- ⚠️ **Use for**: Experimental features
- ⚠️ **Use for**: Development only

### Decision Matrix

| Situation | Use This Model |
|-----------|----------------|
| Production system | Best Model |
| Critical detection | Best Model |
| Just trained new model | Latest Model (test first!) |
| After retraining | Latest Model (validate, then promote) |
| Unknown quality | Best Model (safe choice) |
| Development/testing | Latest Model |

---

## 📋 How to View Model Information

### Method 1: List All Models (Detailed)

```bash
# SSH to Jetson or open terminal on Jetson
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"
```

**Example Output:**
```
================================================================================
AVAILABLE MODELS
================================================================================
v1.3.0
  Created: 2025-11-02T15:45:22.123456
  Trained samples: 1500
  Validation Accuracy: 97.2%
  Precision: 95.8%
  Recall: 96.5%
  F1 Score: 0.961
  
v1.2.0
  Created: 2025-11-02T11:42:08.789667
  Trained samples: 1000
  Validation Accuracy: 96.5%
  Precision: 94.2%
  Recall: 95.8%
  F1 Score: 0.950
  
v1.1.0 👉 CURRENT
  Created: 2025-11-02T10:36:36.878762
  Trained samples: 500
  Validation Accuracy: 92.0%
  Precision: 88.5%
  Recall: 90.0%
  F1 Score: 0.892
================================================================================

🏆 BEST MODEL: v1.3.0 (Accuracy: 97.2%)
📊 CURRENT MODEL: v1.1.0
⚠️  Consider loading v1.3.0 for better performance!
```

### Method 2: Check Current Model Only

```bash
docker logs custom-lstm-detector | grep "CURRENT"
```

**Example Output:**
```
v1.1.0 👉 CURRENT
```

### Method 3: During Training (Real-time)

```bash
# Watch training progress live
docker logs -f custom-lstm-detector
```

**Look for:**
- Epoch progress: `Epoch 10/100: loss=0.123, accuracy=0.95`
- Validation results at end of training
- Model saved confirmation with metrics

### Method 4: Via Operator Interface

1. Open **Operator Interface** in browser
2. Click **"📋 List Available Models"** button
3. Go to **Dashboard** tab
4. Check **"Last System Status"** section
5. Models listed with basic information

---

## 🔍 Example: Comparing Models Step-by-Step

### Step 1: List Available Models

```bash
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"
```

### Step 2: Analyze the Output

Look at this example comparison:

**Model A (v1.2.0):**
- Accuracy: 96.5%
- Precision: 94.2%
- Recall: 95.8%
- F1 Score: 0.950
- Samples: 1000

**Model B (v1.1.0) - Current:**
- Accuracy: 92.0%
- Precision: 88.5%
- Recall: 90.0%
- F1 Score: 0.892
- Samples: 500

### Step 3: Compare Metrics

| Metric | v1.2.0 | v1.1.0 | Difference |
|--------|--------|--------|------------|
| Accuracy | 96.5% | 92.0% | +4.5% ✅ |
| Precision | 94.2% | 88.5% | +5.7% ✅ |
| Recall | 95.8% | 90.0% | +5.8% ✅ |
| F1 Score | 0.950 | 0.892 | +0.058 ✅ |
| Samples | 1000 | 500 | +500 ✅ |

### Step 4: Make Decision

**Winner: v1.2.0** 🏆

**Reasoning:**
- ✅ Higher accuracy (+4.5%)
- ✅ Better precision (fewer false alarms)
- ✅ Better recall (catches more falls)
- ✅ Trained on 2x more data
- ✅ Better F1 score (overall performance)

**Action:** Load v1.2.0 for production use

### Step 5: Load the Better Model

**In Operator Interface:**
1. Go to **Inference Control** tab
2. Click **"🏆 Load Best Model"**
3. System automatically loads v1.2.0
4. Verify in Dashboard tab

**Or load specific version:**
1. Enter "v1.2.0" in version field
2. Click **"🎯 Load Specific Version"**

---

## 🎯 How System Selects "Best Model"

### Selection Algorithm

The system ranks models using this priority:

1. **Primary Criterion**: Validation Accuracy
   - Highest accuracy wins
   - Most important factor

2. **Secondary Criterion**: F1 Score
   - If accuracy is tied
   - Balances precision and recall

3. **Tertiary Criterion**: Training Samples
   - If accuracy and F1 tied
   - More data = more reliable

4. **Final Tiebreaker**: Creation Date
   - If all else equal
   - Most recent wins

### Example Ranking

**Available Models:**
- v1.3.0: Accuracy 97.2%, F1 0.961, 1500 samples → **Rank 1 (BEST)**
- v1.2.0: Accuracy 96.5%, F1 0.950, 1000 samples → **Rank 2**
- v1.1.0: Accuracy 92.0%, F1 0.892, 500 samples → **Rank 3**

**Result:** System recommends v1.3.0 as "Best Model"

---

## 📊 Validation Results During Training

### Where to Find Validation Metrics

After training completes, look for this section in logs:

```bash
docker logs custom-lstm-detector | grep -A 20 "Validation Results"
```

**Example Output:**
```
================================================================================
VALIDATION RESULTS - Model v1.3.0
================================================================================
Dataset: 200 samples (150 normal, 50 falls)

Classification Report:
                  precision    recall  f1-score   support

       Normal       0.978     0.987     0.982       150
         Fall       0.960     0.940     0.950        50

     accuracy                           0.972       200
    macro avg       0.969     0.963     0.966       200
 weighted avg       0.972     0.972     0.972       200

Confusion Matrix:
              Predicted Normal  Predicted Fall
Actual Normal             148                2
Actual Fall                 3               47

Overall Accuracy: 97.2%
Precision: 96.0%
Recall: 94.0%
F1 Score: 0.950

✅ Model saved: /app/models/v1.3.0/
================================================================================
```

### Understanding the Validation Report

**Confusion Matrix:**
```
                Predicted Normal  Predicted Fall
Actual Normal              148               2     ← 2 false alarms
Actual Fall                  3              47     ← 3 missed falls
```

**Interpretation:**
- **True Negatives (148)**: Correctly identified normal activities
- **False Positives (2)**: False alarms (normal predicted as fall)
- **False Negatives (3)**: Missed falls (fall predicted as normal) ⚠️
- **True Positives (47)**: Correctly detected falls

**Key Takeaways:**
- Low false negatives (3) = Good! Missing only 6% of falls
- Low false positives (2) = Good! Only 1.3% false alarm rate
- High accuracy (97.2%) = Excellent overall performance

---

## ⚠️ Red Flags: When NOT to Use a Model

### Don't Use If:

❌ **Recall < 85%**
- Missing too many falls
- DANGEROUS for fall detection
- Retrain with more fall samples

❌ **Accuracy < 85%**
- Too many mistakes overall
- Not reliable for production
- Retrain with more/better data

❌ **Trained on < 500 samples**
- Not enough data
- May not generalize well
- Collect more training data

❌ **No validation performed**
- Unknown performance
- Could be terrible
- Always validate before use

❌ **High false negatives in validation**
- Missing real falls frequently
- Safety critical issue
- Improve training data balance

### Safe Thresholds for Production

✅ **Minimum Requirements:**
- Accuracy: >90%
- Precision: >85%
- Recall: >90% (MOST IMPORTANT)
- F1 Score: >0.90
- Training samples: >500

✅ **Recommended Targets:**
- Accuracy: >95%
- Precision: >90%
- Recall: >95%
- F1 Score: >0.95
- Training samples: >1000

---

## 🚀 Quick Decision Flow

```
Need to load a model?
    │
    ├─ For Production? 
    │   └─ YES → Load "Best Model" 🏆
    │
    ├─ Just trained new model?
    │   └─ YES → Validate first, then compare metrics
    │
    ├─ Testing/Development?
    │   └─ YES → Load "Latest Model" or specific version
    │
    └─ Unsure?
        └─ Load "Best Model" (safe choice) ✅
```

---

## 📝 Commands Cheat Sheet

```bash
# === View Models ===

# List all models with full details
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# Check current model
docker logs custom-lstm-detector | grep "CURRENT"

# Find best model
docker logs custom-lstm-detector | grep "BEST MODEL"

# === View Metrics ===

# Validation results
docker logs custom-lstm-detector | grep -A 20 "Validation Results"

# Specific model metrics
docker logs custom-lstm-detector | grep -A 10 "v1.2.0"

# Training progress
docker logs -f custom-lstm-detector

# === During Training ===

# Watch real-time
docker logs -f custom-lstm-detector

# Last 100 lines
docker logs --tail=100 custom-lstm-detector

# === Compare Models ===

# All models side by side
docker logs custom-lstm-detector | grep -A 5 "v1\." | head -50

# === System Status ===

# Current system state
docker logs custom-lstm-detector | grep "Status published"
```

---

## 🎓 Best Practices

### 1. Always Validate After Training
```bash
# Check validation results immediately
docker logs custom-lstm-detector | tail -50
```

### 2. Compare Before Loading
```bash
# List models and compare metrics
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# Make informed decision based on metrics
```

### 3. Use Best Model for Production
- Don't assume latest = best
- Check metrics first
- Load best performing model

### 4. Monitor Performance Over Time
- Track prediction accuracy
- Note false alarm rates
- Retrain when performance drops

### 5. Document Your Models
```bash
# Create log of model performance
echo "Model v1.3.0 - $(date)" >> model_log.txt
docker logs custom-lstm-detector | grep -A 10 "v1.3.0" >> model_log.txt
```

---

## 📞 Need Help?

### Understanding Metrics
→ See "Understanding Model Performance Metrics" section above

### Choosing Between Models
→ See "Example: Comparing Models Step-by-Step" section

### Loading Best Model
→ Use Operator Interface → Inference Control → "Load Best Model"

### Validation Issues
→ Check OPERATOR_GUIDE.md → Training Workflow → Validation

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**For System**: Fall Detection v2.0

**Quick Start**: Run `docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"` to see all your models!
