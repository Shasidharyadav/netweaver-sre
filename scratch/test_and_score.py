import os
import subprocess
import re

# Use the token from environment
api_key = os.environ.get("HF_TOKEN")

run_env = os.environ.copy()
run_env["API_KEY"] = api_key
run_env["ENV_URL"] = "http://localhost:9000"

print("Starting inference process...")
proc = subprocess.Popen(["python", "inference.py"], 
                        env=run_env, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True, 
                        bufsize=1)

scores = []
for line in proc.stdout:
    print(line, end='')
    # Look for [END] logs
    if "[END]" in line:
        match = re.search(r"task=(\S+).*score=([\d.]+)", line)
        if match:
            scores.append((match.group(1), match.group(2)))

proc.wait()
print("\nFINAL SCORES:")
for t, s in scores:
    print(f"{t}: {s}")
