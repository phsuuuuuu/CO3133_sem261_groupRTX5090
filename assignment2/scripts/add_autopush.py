import json
from pathlib import Path

nb_path = Path("colab_U_net.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

# 1. Add GitHub Token fields to Cell 1 (CONFIGURATION PANEL)
cell1_lines = nb["cells"][1]["source"]
new_c1 = []
for line in cell1_lines:
    if "### 1. Cityscapes account" in line:
        new_c1.append("# @markdown ### 0. GitHub Auto-Push (Optional - Automatically commit & push logs to GitHub)\n")
        new_c1.append("GITHUB_TOKEN = \"\"  # @param {type:\"string\"}\n")
        new_c1.append("GIT_USER_NAME = \"Cao Duc Thanh\"  # @param {type:\"string\"}\n")
        new_c1.append("GIT_USER_EMAIL = \"thanh.caothanh4225@hcmut.edu.vn\"  # @param {type:\"string\"}\n")
        new_c1.append("# @markdown --- \n")
    new_c1.append(line)

nb["cells"][1]["source"] = new_c1

# 2. Add auto_push_to_github helper function into Cell 5 (source code)
cell5_text = "".join(nb["cells"][5]["source"])
github_helper = """
# ============================== GitHub Auto-Push Helper ==============================
import subprocess

def push_results_to_github(commit_msg="Update U-Net baseline training logs and test metrics"):
    token = globals().get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = ""
    if not token:
        print("[GITHUB] GITHUB_TOKEN is not provided. Skipping auto-push.")
        print("          You can save checkpoints to Google Drive or download them locally.")
        return False

    user_name = globals().get("GIT_USER_NAME", "Cao Duc Thanh")
    user_email = globals().get("GIT_USER_EMAIL", "thanh.caothanh4225@hcmut.edu.vn")
    repo_url = f"https://{token}@github.com/phsuuuuuu/CO3133_sem261_groupRTX5090.git"

    print(f"\\n[GITHUB] Preparing to push results to GitHub as {user_name}...")
    try:
        subprocess.run(["git", "config", "user.name", user_name], check=True)
        subprocess.run(["git", "config", "user.email", user_email], check=True)
        
        # Pull latest changes to avoid conflicts
        subprocess.run(["git", "pull", "--rebase", repo_url, "main"], check=False)

        # Stage only lightweight results: log.csv and metrics_*.json (NEVER stage heavy .pt weights)
        staged_any = False
        for p in CKPT_ROOT.rglob("*"):
            if p.is_file() and (p.name == "log.csv" or p.name.startswith("metrics_")):
                rel_path = str(p.relative_to(WORK_DIR.parent if (WORK_DIR / ".git").exists() else WORK_DIR))
                subprocess.run(["git", "add", str(p)], check=True)
                staged_any = True
                print(f"   + Staged: {p.relative_to(CKPT_ROOT)}")

        if not staged_any:
            print("[GITHUB] No new log.csv or metrics_*.json found to push.")
            return False

        commit_res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if commit_res.returncode != 0 and "nothing to commit" in commit_res.stdout.lower():
            print("[GITHUB] Everything up to date — nothing new to commit.")
            return True

        push_res = subprocess.run(["git", "push", repo_url, "main"], capture_output=True, text=True)
        if push_res.returncode == 0:
            print("[GITHUB] SUCCESS: Training logs & test metrics pushed to GitHub repository!")
            return True
        else:
            print(f"[GITHUB] ERROR: Push failed:\\n{push_res.stderr}")
            return False
    except Exception as e:
        print(f"[GITHUB] Exception during push: {e}")
        return False
"""

cell5_text = cell5_text + "\n" + github_helper
nb["cells"][5]["source"] = [l + "\n" for l in cell5_text.split("\n")[:-1]] + [cell5_text.split("\n")[-1]]

# 3. Add auto-push right after U-Net training (Cell 17)
cell17_lines = nb["cells"][17]["source"]
cell17_lines.append("    # Auto-push logs immediately after training completes\n")
cell17_lines.append("    push_results_to_github('Colab: update U-Net baseline training log.csv')\n")
nb["cells"][17]["source"] = cell17_lines

# 4. Insert an explicit GitHub Push Cell right after Step 8 (Evaluation & Summary)
push_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Step 8.1: Push logs and test metrics to GitHub repository\n",
        "# Automatically commits and pushes log.csv & metrics_test.json (keeps .pt weights on Drive/local)\n",
        "push_results_to_github('Colab: upload final U-Net baseline test metrics_test.json and logs')\n"
    ]
}

# Find index of summary table cell (cell 23)
summary_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code" and "# @title Summary table" in "".join(c["source"]):
        summary_idx = i
        break

if summary_idx is not None:
    nb["cells"].insert(summary_idx + 1, push_cell)

with open("colab_U_net.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated colab_U_net.ipynb with auto-push to GitHub feature!")
