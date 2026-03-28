from daytona import Daytona, DaytonaConfig
import os
from dotenv import load_dotenv

# Load .env from the parent directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Fallback to current directory
    load_dotenv()

daytona_api = os.getenv("DAYTONA_API_KEY")
if not daytona_api:
    raise RuntimeError("DAYTONA_API_KEY is not set. Please set it in your .env file or environment variables.")

config = DaytonaConfig(api_key=daytona_api)
daytona = Daytona(config)


sandboxes = daytona.list()
items = sandboxes.items if hasattr(sandboxes, 'items') else list(sandboxes)
print(f"Found {len(items)} sandbox(es).")

for s in items:
    print(f"Deleting sandbox: {s.id}")
    daytona.delete(s)

print("All sandboxes deleted.")