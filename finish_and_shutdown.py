import time
import os
import subprocess
import sys

log_file = r'C:\Users\karthikkrazy\.gemini\antigravity\brain\59c57cfd-93ed-45e7-bd9a-e9df181440ef\.system_generated\tasks\task-1875.log'
repo_dir = r'C:\Users\karthikkrazy\Documents\antigravity\vibrant-kepler'

print('Waiting for 200-epoch SmolVLA training to complete...', flush=True)

while True:
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if '[SUCCESS] SmolVLA Model Checkpoint saved' in content or '200/200' in content:
                print('SmolVLA training has finished successfully!', flush=True)
                break
    time.sleep(15)

# 1. Add all changes to git and push
print('Adding smolvla_dobot to git...', flush=True)
subprocess.run(['git', 'add', 'smolvla_dobot/'], cwd=repo_dir, check=True)
print('Committing...', flush=True)
subprocess.run(['git', 'commit', '-m', 'Add smolvla_dobot architecture with ViT patch encoder, multimodal self/cross attention, and 200-epoch weights'], cwd=repo_dir, check=True)
print('Pushing to GitHub...', flush=True)
subprocess.run(['git', 'push', 'origin', 'master'], cwd=repo_dir, check=True)
print('GitHub sync complete!', flush=True)

# 2. Shutdown PC
print('Initiating shutdown -s -t 30...', flush=True)
subprocess.run(['shutdown', '/s', '/t', '30'], check=True)
