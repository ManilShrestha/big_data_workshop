# OpenAI Batch API Workflow

This guide explains how to use the batch mode without waiting 24 hours to verify your code works.

## The Problem

OpenAI's Batch API requires a 24h completion window, but batches typically complete in minutes to hours. You don't want to wait potentially 24 hours just to find out if your code has a bug!

## The Solution: Dry-Run Mode

Use `--dry-run` to create the batch and verify everything works, then come back later to retrieve results.

---

## Quick Start

### 1. Test Batch Creation (Immediate Feedback)

```bash
# Create a batch with 5 questions and get the batch ID immediately
python variant0_llm_baseline.py --mode batch --limit 5 --dry-run
```

**What this does:**
- ✅ Loads your questions
- ✅ Creates the batch request
- ✅ Uploads to OpenAI
- ✅ Returns batch ID immediately
- ❌ Does NOT wait for completion

**Output:**
```
================================================================================
DRY-RUN MODE: Batch created successfully!
Batch ID: batch_abc123xyz
================================================================================

To resume and wait for this batch, run:
  python variant0_llm_baseline.py --mode batch --batch-id batch_abc123xyz

To check batch status manually, use:
  python batch_manager.py --status batch_abc123xyz
================================================================================
```

### 2. Check Batch Status (Anytime)

```bash
# Check if your batch is ready
python batch_manager.py --status batch_abc123xyz
```

**Possible statuses:**
- `validating` - OpenAI is validating the request
- `in_progress` - Processing your questions
- `finalizing` - Almost done
- `completed` - ✅ Ready to retrieve!
- `failed` - ❌ Something went wrong

### 3. List All Your Batches

```bash
# See all recent batches
python batch_manager.py --list

# See more batches
python batch_manager.py --list --limit 20
```

### 4. Retrieve Results (Once Complete)

```bash
# Resume the batch and wait for it to complete, then process results
python variant0_llm_baseline.py --mode batch --batch-id batch_abc123xyz
```

This will:
- Poll for status every 30s (default)
- Download results when ready
- Process and evaluate answers
- Save to JSON file

### 5. Cancel a Batch (If Needed)

```bash
python batch_manager.py --cancel batch_abc123xyz
```

---

## Full Workflow Example

### Scenario: You want to test batch mode on 1-hop questions

**Step 1: Create batch (dry-run)**
```bash
python variant0_llm_baseline.py --mode batch --limit 10 --dry-run
```
Output: `batch_abc123xyz`

**Step 2: Verify it was created**
```bash
python batch_manager.py --status batch_abc123xyz
```

**Step 3: Go do other work** ☕
- The batch processes in the background
- Typically completes in 5-30 minutes for small batches

**Step 4: Check status later**
```bash
python batch_manager.py --status batch_abc123xyz
```

If status = `completed`:

**Step 5: Retrieve and process results**
```bash
python variant0_llm_baseline.py --mode batch --batch-id batch_abc123xyz
```

Results saved to: `results/variant0_llm_baseline_1-hop-test_batch.json`

---

## Comparison: Direct vs Batch Mode

| Feature | Direct Mode | Batch Mode |
|---------|-------------|------------|
| **Cost** | Full price | 50% cheaper |
| **Speed** | Immediate | 5min - 24h |
| **Use Case** | Testing, debugging | Production, large datasets |
| **Command** | `--mode direct` | `--mode batch --dry-run` |

---

## Common Commands

```bash
# Quick test with direct mode (immediate results)
python variant0_llm_baseline.py --mode direct --limit 5

# Create batch for testing (get ID, don't wait)
python variant0_llm_baseline.py --mode batch --limit 5 --dry-run

# Check batch status
python batch_manager.py --status <BATCH_ID>

# Resume and retrieve results
python variant0_llm_baseline.py --mode batch --batch-id <BATCH_ID>

# List all batches
python batch_manager.py --list

# Multiple datasets in batch mode
python variant0_llm_baseline.py --mode batch --datasets 1-hop 2-hop --limit 10 --dry-run
```

---

## Tips

1. **Start with --dry-run**: Always use dry-run mode first to verify your batch was created successfully
2. **Use --limit for testing**: Test with `--limit 5` before running full datasets
3. **Poll interval**: Use `--poll-interval 60` for longer intervals if you don't want to be checking constantly
4. **Check status manually**: Use `batch_manager.py` to check status without starting a long-running poll process

---

## Troubleshooting

**Batch stays in "validating" for a long time**
- This is normal. OpenAI validates all requests before processing
- Usually takes 1-5 minutes

**Batch failed**
- Use `batch_manager.py --status <BATCH_ID>` to see error details
- Common issues: malformed JSON, API rate limits

**I lost my batch ID**
- Run `python batch_manager.py --list` to see recent batches
- Look for the description matching your dataset

**Want to test code without using API**
- Use direct mode with very small limit: `--mode direct --limit 1`
