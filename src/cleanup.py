from daytona import Daytona, DaytonaConfig
import os

DaytonaConfig.load_from_env()
daytona = Daytona()

sandboxes = daytona.list()
items = sandboxes.items if hasattr(sandboxes, 'items') else list(sandboxes)
print(f"Found {len(items)} sandbox(es).")

for s in items:
    print(f"Deleting sandbox: {s.id}")
    daytona.delete(s)

print("All sandboxes deleted.")
