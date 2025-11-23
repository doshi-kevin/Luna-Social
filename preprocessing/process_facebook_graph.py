import os
import pandas as pd
import networkx as nx
from pathlib import Path

# Configuration
FACEBOOK_GRAPH_DIR = "datasets/facebook_graph/facebook"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "social_graph.edgelist")

def create_output_directory():
    """Create processed output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def find_edge_files():
    """Discover all .edges files in the facebook graph directory."""
    print("[*] Scanning for .edges files...")
    
    if not os.path.exists(FACEBOOK_GRAPH_DIR):
        raise FileNotFoundError(f"Facebook graph directory not found: {FACEBOOK_GRAPH_DIR}")
    
    edge_files = []
    for file in os.listdir(FACEBOOK_GRAPH_DIR):
        if file.endswith('.edges'):
            full_path = os.path.join(FACEBOOK_GRAPH_DIR, file)
            edge_files.append(full_path)
    
    print(f"[+] Found {len(edge_files)} .edges files")
    for f in sorted(edge_files):
        print(f"   - {os.path.basename(f)}")
    
    if not edge_files:
        raise FileNotFoundError("No .edges files found in facebook graph directory")
    
    return sorted(edge_files)

def load_single_edges_file(filepath):
    """Load a single .edges file with format: source_node target_node."""
    print(f"[*] Loading {os.path.basename(filepath)}...")
    
    edges = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse edge: format is typically "node1 node2"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node1 = int(parts[0])
                        node2 = int(parts[1])
                        edges.append((node1, node2))
                    except ValueError:
                        print(f"   [!] Warning: Could not parse line {line_num}: {line}")
                        continue
    
    except Exception as e:
        print(f"   [!] Error reading file: {str(e)}")
        return []
    
    print(f"   [+] Loaded {len(edges)} edges")
    return edges

def combine_edge_lists(edge_files):
    """Combine all edges from multiple files into single list."""
    print("[*] Combining edge lists...")
    
    all_edges = []
    edge_count_by_file = {}
    
    for filepath in edge_files:
        edges = load_single_edges_file(filepath)
        file_name = os.path.basename(filepath)
        edge_count_by_file[file_name] = len(edges)
        all_edges.extend(edges)
    
    print(f"[+] Total edges before deduplication: {len(all_edges)}")
    
    return all_edges, edge_count_by_file

def remove_duplicate_edges(edges):
    """Remove duplicate and self-loop edges from edge list."""
    print("[*] Removing duplicates and self-loops...")
    
    initial_count = len(edges)
    
    # Remove self-loops
    edges = [e for e in edges if e[0] != e[1]]
    self_loops_removed = initial_count - len(edges)
    
    # Remove duplicates while preserving edge direction
    # For undirected graphs, we might also want to normalize (a,b) == (b,a)
    edges_set = set(edges)
    duplicates_removed = len(edges) - len(edges_set)
    
    edges = list(edges_set)
    
    print(f"[+] Self-loops removed: {self_loops_removed}")
    print(f"[+] Duplicate edges removed: {duplicates_removed}")
    print(f"[+] Final edge count: {len(edges)}")
    
    return edges

def compute_graph_statistics(edges):
    """Compute basic graph statistics."""
    print("[*] Computing graph statistics...")
    
    # Build networkx graph for analysis
    G = nx.Graph()
    G.add_edges_from(edges)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Find connected components
    num_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_size = len(largest_cc)
    largest_cc_fraction = largest_cc_size / num_nodes if num_nodes > 0 else 0
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {num_edges}")
    print(f"   Density: {density:.6f}")
    print(f"   Connected components: {num_components}")
    print(f"   Largest component: {largest_cc_size} nodes ({largest_cc_fraction:.2%})")
    print(f"   Avg degree: {avg_degree:.2f}")
    
    return G

def save_edgelist(edges, output_path):
    """Save edge list to edgelist format."""
    print(f"[*] Saving edgelist to {output_path}...")
    
    # Create DataFrame for structured save
    df_edges = pd.DataFrame(edges, columns=['source', 'target'])
    df_edges = df_edges.drop_duplicates()
    
    # Write as space-separated edgelist format
    with open(output_path, 'w') as f:
        f.write("# source target\n")
        for _, row in df_edges.iterrows():
            f.write(f"{row['source']} {row['target']}\n")
    
    print(f"[+] Saved {len(df_edges)} edges to edgelist")
    
    return df_edges

def validate_edgelist(output_path):
    """Validate saved edgelist file integrity."""
    print("[*] Validating edgelist file...")
    
    edge_count = 0
    try:
        with open(output_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    edge_count += 1
    except Exception as e:
        print(f"   [!] Validation failed: {str(e)}")
        return False
    
    print(f"[+] Validation passed: {edge_count} valid edges in file")
    return True

def main():
    """Main execution pipeline."""
    print("="*60)
    print("FACEBOOK GRAPH PREPROCESSING PIPELINE")
    print("="*60)
    
    try:
        create_output_directory()
        
        # Find all edge files
        edge_files = find_edge_files()
        
        # Combine edges from all files
        all_edges, edge_count_by_file = combine_edge_lists(edge_files)
        
        # Remove duplicates and self-loops
        clean_edges = remove_duplicate_edges(all_edges)
        
        # Compute graph statistics
        graph = compute_graph_statistics(clean_edges)
        
        # Save edgelist
        df_edges = save_edgelist(clean_edges, OUTPUT_FILE)
        
        # Validate output
        validate_edgelist(OUTPUT_FILE)
        
        print("="*60)
        print(f"✓ COMPLETE: social_graph.edgelist created")
        print(f"  Location: {OUTPUT_FILE}")
        print(f"  Edges: {len(df_edges)}")
        print("="*60)
        
    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()