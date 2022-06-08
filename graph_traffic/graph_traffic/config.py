from pathlib import Path
import os

project_path = Path(r"C:\Users\Bened\PycharmProjects\madrid-traffic")
data_path = os.path.join(project_path, "data")
figures_path = f"{project_path}/figures"
training_path = f"{project_path}/training_history"