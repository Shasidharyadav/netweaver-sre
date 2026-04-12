import os
import subprocess
from dotenv import load_dotenv

# Try to load .env
load_dotenv()

# Extract from .env manually if needed (handling the space in 'hf token')
env_vars = {}
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                env_vars[k.strip().replace(' ', '_').upper()] = v.strip()

# Priority: Environment > Manual Parsing
api_key = os.getenv("API_KEY") or env_vars.get("HF_TOKEN") or env_vars.get("API_KEY")

if not api_key:
    print("CRITICAL: No API_KEY found in .env or environment.")
    exit(1)

# Set environment variables for the sub-process
run_env = os.environ.copy()
run_env["API_KEY"] = api_key
run_env["ENV_URL"] = "http://localhost:8000" # Test local server

print(f"Starting inference against local server...")
try:
    result = subprocess.run(["python", "inference.py"], env=run_env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
except Exception as e:
    print(f"Execution failed: {e}")
