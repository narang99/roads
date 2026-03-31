import argparse
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil
from collections import Counter

def load_data(embs_path, paths_file=None):
    data = torch.load(embs_path)
    labels_val = None
    if isinstance(data, dict):
        embs = data['embs']
        paths = data['paths']
        labels_val = data.get('labels')
    else:
        embs = data
        if paths_file is None:
            raise ValueError("If embs is a tensor, paths_file must be provided.")
        with open(paths_file, 'r') as f:
            paths = [line.strip() for line in f.readlines()]
    
    if len(embs) != len(paths):
        raise ValueError(f"Mismatch: {len(embs)} embeddings and {len(paths)} paths.")
    
    return embs, paths, labels_val

def cluster_embeddings(embs, n_clusters, n_pca=50):
    # Normalize
    X = F.normalize(embs, p=2, dim=1).numpy()
    
    # PCA
    pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0]), random_state=42)
    X_pca = pca.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    return labels

def create_thumbnail(img_path, out_path, size=(224, 224)):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        img_res = cv2.resize(img, size)
        cv2.imwrite(str(out_path), img_res)
        return True
    except Exception:
        return False

def generate_report(embs, paths, labels_val, k_range, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)
    
    print("Generating thumbnails...")
    rel_thumbnails = []
    for i, p in enumerate(tqdm(paths)):
        thumb_name = f"thumb_{i}.jpg"
        thumb_path = thumbnails_dir / thumb_name
        if not thumb_path.exists():
            if not create_thumbnail(p, thumb_path):
                # fallback or skip
                pass
        rel_thumbnails.append(f"thumbnails/{thumb_name}")

    index_html = """
    <html>
    <head>
        <title>K-Means Analysis</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
            h1 { color: #333; }
            ul { list-style: none; padding: 0; }
            li { background: white; margin: 10px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            a { text-decoration: none; color: #007bff; font-weight: bold; }
            a:hover { color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>K-Means Clustering Analysis</h1>
        <ul>
    """
    
    for k in k_range:
        print(f"Clustering for K={k}...")
        cluster_labels = cluster_embeddings(embs, k)
        
        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(exist_ok=True)
        
        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
            
        # Generate K-specific index
        k_index_html = f"""
        <html>
        <head>
            <title>K={k}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f9f9f9; }}
                .thumb-container {{ display: flex; gap: 5px; overflow-x: auto; }}
                img {{ border-radius: 4px; border: 1px solid #eee; object-fit: cover; }}
            </style>
        </head>
        <body>
            <h1>Clusters for K={k}</h1>
            <a href="../index.html">← Back to All K</a>
            <table>
                <tr><th>Cluster ID</th><th>Size</th>{"<th>Purity/Main Class</th>" if labels_val is not None else ""}<th>Samples</th></tr>
        """
        
        for cid, c_indices in enumerate(clusters):
            accuracy_str = ""
            if labels_val is not None:
                c_labels = [labels_val[i] for i in c_indices]
                if len(c_labels) > 0:
                    most_common = Counter(c_labels).most_common(1)[0]
                    acc = most_common[1] / len(c_labels)
                    accuracy_str = f"<td>{acc:.2f} (class {most_common[0]})</td>"
                else:
                    accuracy_str = "<td>-</td>"

            cluster_page_name = f"cluster_{cid}.html"
            k_index_html += f"<tr><td><a href='{cluster_page_name}'>{cid}</a></td><td>{len(c_indices)}</td>{accuracy_str}<td><div class='thumb-container'>"
            
            # Show top 8 thumbnails
            for idx in c_indices[:8]:
                k_index_html += f"<img src='../{rel_thumbnails[idx]}' width='100' height='100'>"
            k_index_html += "</div></td></tr>"
            
            # Generate Cluster page
            cluster_html = f"""
            <html>
            <head>
                <title>K={k} Cluster={cid}</title>
                <style>
                    body {{ font-family: sans-serif; margin: 20px; }}
                    .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
                    .item {{ width: 220px; text-align: center; border: 1px solid #ddd; padding: 5px; border-radius: 5px; }}
                    img {{ max-width: 100%; border-radius: 3px; }}
                    small {{ color: #666; display: block; margin-top: 5px; overflow-wrap: break-word; }}
                </style>
            </head>
            <body>
                <h1>K={k} Cluster={cid} (Size: {len(c_indices)})</h1>
                <a href='index.html'>← Back to K={k} List</a>
                <hr>
                <div class='grid'>
            """
            for idx in c_indices:
                label_info = f"<br>Label: {labels_val[idx]}" if labels_val is not None else ""
                cluster_html += f"<div class='item'><img src='../{rel_thumbnails[idx]}'><small>{Path(paths[idx]).name}{label_info}</small></div>"
            cluster_html += "</div></body></html>"
            
            with open(k_dir / cluster_page_name, 'w') as f:
                f.write(cluster_html)
                
        k_index_html += "</table><br><a href='../index.html'>Back to Index</a></body></html>"
        with open(k_dir / "index.html", 'w') as f:
            f.write(k_index_html)
            
        index_html += f"<li><a href='k_{k}/index.html'>K = {k}</a> &mdash; {k} clusters, {len(paths)} images</li>"

    index_html += "</ul></body></html>"
    with open(output_dir / "index.html", 'w') as f:
        f.write(index_html)

    print(f"Report generated at {output_dir}/index.html")

def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings using K-Means and generate HTML report.")
    parser.add_argument("--embs", required=True, help="Path to embeddings .pt file (tensor or dict with 'embs', 'paths', and optional 'labels')")
    parser.add_argument("--paths", help="Path to text file with image paths (one per line) if .pt is just a tensor")
    parser.add_argument("--k-start", type=int, default=10)
    parser.add_argument("--k-end", type=int, default=50)
    parser.add_argument("--k-step", type=int, default=10)
    parser.add_argument("--out", default="reports/clusters", help="Output directory")
    
    args = parser.parse_args()
    
    embs, paths, labels_val = load_data(args.embs, args.paths)
    
    k_range = range(args.k_start, args.k_end + 1, args.k_step)
    
    generate_report(embs, paths, labels_val, k_range, args.out)

if __name__ == "__main__":
    main()
