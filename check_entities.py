#!/usr/bin/env python3
"""
Script to check if all entities in test.txt are present in entity2id.txt
"""

def load_entities_from_entity2id(filepath):
    """Load all entity names from entity2id.txt"""
    entities = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by tab or whitespace
                parts = line.split()
                if parts:
                    entity = parts[0]
                    entities.add(entity)
    return entities

def extract_entities_from_triples(filepath):
    """Extract all entities (head and tail) from a triples file"""
    entities = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    head_entity = parts[0]
                    tail_entity = parts[2]
                    entities.add(head_entity)
                    entities.add(tail_entity)
    return entities

def main():
    # File paths
    entity2id_path = 'data/fb15k237/entity2id.txt'
    train_path = 'data/fb15k237/train.txt'
    valid_path = 'data/fb15k237/valid.txt'
    test_path = 'data/fb15k237/test.txt'
    
    print("Loading entities from entity2id.txt...")
    known_entities = load_entities_from_entity2id(entity2id_path)
    print(f"Total entities in entity2id.txt: {len(known_entities)}")
    
    # Extract entities from each file
    print("\nExtracting entities from train.txt...")
    train_entities = extract_entities_from_triples(train_path)
    print(f"Total unique entities in train.txt: {len(train_entities)}")
    
    print("\nExtracting entities from valid.txt...")
    valid_entities = extract_entities_from_triples(valid_path)
    print(f"Total unique entities in valid.txt: {len(valid_entities)}")
    
    print("\nExtracting entities from test.txt...")
    test_entities = extract_entities_from_triples(test_path)
    print(f"Total unique entities in test.txt: {len(test_entities)}")
    
    # Combine all entities
    all_triples_entities = train_entities | valid_entities | test_entities
    print(f"\nTotal unique entities across all files: {len(all_triples_entities)}")
    
    # Check for missing entities
    print("\nChecking for missing entities...")
    missing_entities = all_triples_entities - known_entities
    
    if missing_entities:
        print(f"\n❌ Found {len(missing_entities)} missing entities in entity2id.txt:")
        print("-" * 50)
        for entity in sorted(missing_entities)[:20]:  # Show first 20
            print(f"  - {entity}")
        if len(missing_entities) > 20:
            print(f"  ... and {len(missing_entities) - 20} more")
    else:
        print("\n✅ All entities in train/valid/test are present in entity2id.txt!")
    
    # Check each file individually
    print("\n" + "=" * 50)
    print("INDIVIDUAL FILE CHECKS:")
    print("=" * 50)
    
    for name, entities in [("train.txt", train_entities), 
                           ("valid.txt", valid_entities), 
                           ("test.txt", test_entities)]:
        missing = entities - known_entities
        if missing:
            print(f"❌ {name}: {len(missing)} missing entities")
        else:
            print(f"✅ {name}: All entities present")
    
    # Additional statistics
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(f"Entities in entity2id.txt:       {len(known_entities)}")
    print(f"Entities in train.txt:           {len(train_entities)}")
    print(f"Entities in valid.txt:           {len(valid_entities)}")
    print(f"Entities in test.txt:            {len(test_entities)}")
    print(f"Total unique entities (all):     {len(all_triples_entities)}")
    print(f"Missing entities:                {len(missing_entities)}")
    if len(all_triples_entities) > 0:
        print(f"Coverage:                        {(1 - len(missing_entities)/len(all_triples_entities))*100:.2f}%")

if __name__ == "__main__":
    main()

