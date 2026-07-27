import os
import json
from pathlib import Path
import bot

EXPANDED_DIR = Path("dataset/expanded")
test_pairs_path = EXPANDED_DIR / "test_pairs.json"
output_path = Path("submission.jsonl")

print(f"Reading test pairs from {test_pairs_path}")
with open(test_pairs_path, "r", encoding="utf-8") as f:
    test_pairs = json.load(f)["pairs"]

print(f"Loaded {len(test_pairs)} test pairs.")
lines = []

for idx, pair in enumerate(test_pairs):
    test_id = pair["test_id"]
    trigger_id = pair["trigger_id"]
    merchant_id = pair["merchant_id"]
    customer_id = pair.get("customer_id")
    
    print(f"[{idx+1}/30] Processing {test_id} (Trigger: {trigger_id}, Merchant: {merchant_id})")
    
    # Load Trigger
    with open(EXPANDED_DIR / "triggers" / f"{trigger_id}.json", "r", encoding="utf-8") as f:
        trigger = json.load(f)
        
    # Load Merchant
    with open(EXPANDED_DIR / "merchants" / f"{merchant_id}.json", "r", encoding="utf-8") as f:
        merchant = json.load(f)
        
    # Load Category
    cat_slug = merchant["category_slug"]
    with open(EXPANDED_DIR / "categories" / f"{cat_slug}.json", "r", encoding="utf-8") as f:
        category = json.load(f)
        
    # Load Customer (optional)
    customer = None
    if customer_id:
        with open(EXPANDED_DIR / "customers" / f"{customer_id}.json", "r", encoding="utf-8") as f:
            customer = json.load(f)
            
    # Compose message using bot's compose function
    res = bot.compose(category, merchant, trigger, customer)
    
    # Format according to submission schema
    submission_entry = {
        "test_id": test_id,
        "body": res["body"],
        "cta": res["cta"],
        "send_as": res["send_as"],
        "suppression_key": res["suppression_key"],
        "rationale": res["rationale"]
    }
    
    lines.append(json.dumps(submission_entry, ensure_ascii=False))

print(f"Writing output to {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("Submission JSONL generated successfully!")
