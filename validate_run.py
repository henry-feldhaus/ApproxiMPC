import json
import os
import numpy as np
import glob

run_dir = "outputs/datasets/multimap_training/run_20260420T214055Z"
manifest_path = os.path.join(run_dir, "manifest.json")

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

pass_all = True
results = []

for combo in manifest['combos']:
    rel_output_dir = combo['output_dir'].replace('/app/', '')
    combo_id = f"{combo['map']}_{combo['direction']}"
    
    npz_path = os.path.join(rel_output_dir, f"stmpc_{combo_id}.npz")
    json_path = os.path.join(rel_output_dir, f"stmpc_{combo_id}.json")
    
    status = combo['status']
    succ = combo['successful_episodes']
    target = combo['target_episodes']
    
    unique_ids = 0
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        # Use episode_ids instead of episode_id
        if 'episode_ids' in data:
            unique_ids = len(np.unique(data['episode_ids']))
        elif 'episode_id' in data:
            unique_ids = len(np.unique(data['episode_id']))
    
    collisions = -1
    boundaries = -1
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
            collisions = meta.get('num_collision_steps', -1)
            boundaries = meta.get('num_boundary_steps', -1)
            
    attempt_files = glob.glob(os.path.join(rel_output_dir, "*.attempt_*.npz")) + \
                    glob.glob(os.path.join(rel_output_dir, "*.attempt_*.json"))
    num_attempt_files = len(attempt_files)
    
    combo_pass = (status == 'ok' and 
                  succ == target and 
                  unique_ids == target and 
                  collisions == 0 and 
                  boundaries == 0 and 
                  num_attempt_files == 0)
    
    if not combo_pass:
        pass_all = False
        
    results.append({
        'combo': combo_id,
        'status': status,
        'succ_target': f"{succ}/{target}",
        'unique_ids': unique_ids,
        'collisions': collisions,
        'boundaries': boundaries,
        'attempt_files': num_attempt_files,
        'pass': combo_pass
    })

for r in results:
    print(f"Combo: {r['combo']}")
    print(f"  Status: {r['status']}, Succ/Target: {r['succ_target']}, Unique IDs: {r['unique_ids']}")
    print(f"  Collisions: {r['collisions']}, Boundaries: {r['boundaries']}, Attempt Files: {r['attempt_files']}")
    print(f"  PASS: {r['pass']}")

if pass_all:
    print("\nFINAL STATUS: PASS")
else:
    print("\nFINAL STATUS: FAIL")
