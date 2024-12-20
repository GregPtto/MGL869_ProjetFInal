import os
import re
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd

# Chemins et configurations
REPO_PATH = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\hive"
OUTPUT_DIR = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\Projet_FInal\versions2"
BUG_KEYWORDS = ["fix", "fixed", "resolved", "bug", "patch"]
TAG_PATTERN = r".*-(2|3)\.[0-9]\.0$"

def git_command(cmd, cwd):
    """Exécute une commande Git."""
    return subprocess.check_output(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace").strip()

def collect_tags():
    """Récupère et trie les tags en ordre chronologique, en excluant ceux contenant 'storage'."""
    tags = git_command(["git", "tag", "--list"], cwd=REPO_PATH).split("\n")
    valid_tags = [tag for tag in tags if re.match(TAG_PATTERN, tag) and "storage" not in tag and "rel/standalone-metastore-release-3.0.0" not in tag]
    return sorted(valid_tags, key=lambda t: int(git_command(["git", "log", "-1", "--format=%ct", t], cwd=REPO_PATH)))

def parse_diff(commit_hash):
    """Analyse les changements d'un commit."""
    diff_output = git_command(["git", "show", "--numstat", "--pretty=format:", commit_hash], cwd=REPO_PATH).strip()
    return [
        {"file": line.split("\t")[2], "added": int(line.split("\t")[0]), "deleted": int(line.split("\t")[1])}
        for line in diff_output.split("\n")
        if line.endswith((".java", ".cpp", ".c", ".h"))
    ]

def calculate_expertise(developers, cumulative_developers):
    """Calcule l'expertise moyenne et minimale des développeurs."""
    expertise = {dev: cumulative_developers.count(dev) for dev in developers}
    return {
        "mean_expertise": np.mean(list(expertise.values())) if expertise else 0,
        "min_expertise": min(expertise.values()) if expertise else 0,
    }

def calculate_metrics(tag, previous_tag, cumulative_data):
    """Calcule toutes les métriques pour une version donnée."""
    print(f"Collecte des métriques de la version {tag} (précédente version {previous_tag})")
    commits = git_command(["git", "log", "--pretty=format:%H|%an|%at|%s", f"{previous_tag}..{tag}"], cwd=REPO_PATH).split("\n")
    metrics = defaultdict(lambda: defaultdict(list))

    for commit in commits:
        commit_hash, author, timestamp, message = commit.split("|", 3)
        timestamp = int(timestamp)
        is_bug_fix = any(keyword in message.lower() for keyword in BUG_KEYWORDS)

        diff_changes = parse_diff(commit_hash)
        comment_changes = git_command(["git", "show", "-U0", commit_hash], cwd=REPO_PATH)

        for change in diff_changes:
            file = change["file"]
            metrics[file]["added_lines"].append(change["added"])
            metrics[file]["deleted_lines"].append(change["deleted"])
            metrics[file]["developers"].append(author)
            metrics[file]["timestamps"].append(timestamp)
            if is_bug_fix:
                metrics[file]["bug_fix_commits"].append(1)
            if re.search(r"^\+.*//|^\-.*//", comment_changes, re.MULTILINE):
                metrics[file]["comments_changed"].append(1)
            else:
                metrics[file]["comments_unchanged"].append(1)

    results = []
    for file, data in metrics.items():
        timestamps = sorted(data["timestamps"])
        cumulative_data["developers"][file].extend(data["developers"])
        cumulative_data["timestamps"][file].extend(data["timestamps"])
        expertise = calculate_expertise(data["developers"], cumulative_data["developers"][file])

        results.append({
            "file": file,
            "added_lines": sum(data["added_lines"]),
            "deleted_lines": sum(data["deleted_lines"]),
            "commits": len(data["timestamps"]),
            "bug_fix_commits": len(data["bug_fix_commits"]),
            "developers": len(set(data["developers"])),
            "global_developers": len(set(cumulative_data["developers"][file])),
            "average_time_between_changes": np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0,
            "global_average_time_between_changes": np.mean(np.diff(sorted(cumulative_data["timestamps"][file]))) if len(cumulative_data["timestamps"][file]) > 1 else 0,
            "mean_expertise": expertise["mean_expertise"],
            "min_expertise": expertise["min_expertise"],
            "comments_changed": len(data["comments_changed"]),
            "comments_unchanged": len(data["comments_unchanged"]),
        })

    return results

def save_to_csv(version, results):
    """Sauvegarde les résultats dans un CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_version = version.replace("/", "-")  # Remplacer les '/' par des '-'
    output_file = os.path.join(OUTPUT_DIR, f"{metrics_version}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Résultats sauvegardés dans {output_file}.")

if __name__ == "__main__":
    tags = collect_tags()
    cumulative_data = {"developers": defaultdict(list), "timestamps": defaultdict(list)}

    print(tags)


    previous_tag = 'release-1.2.1'
    for tag in tags[0:]:
        metrics = calculate_metrics(tag, previous_tag, cumulative_data)
        save_to_csv(tag, metrics)
        previous_tag = tag 


