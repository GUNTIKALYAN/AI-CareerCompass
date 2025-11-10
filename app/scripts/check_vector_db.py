import os
import chromadb

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "db", "chroma"))

def inspect_collection(path: str):
    """Inspect a Chroma collection inside a given path."""
    if not os.path.exists(path):
        print(f"[!] Directory not found: {path}")
        return

    print(f"\n🔹 Inspecting: {path}")

    try:
        client = chromadb.PersistentClient(path)
        collections = client.list_collections()

        if not collections:
            print("  No collections found.")
            return

        for c in collections:
            print(f"  Collection name: {c.name}")
            collection = client.get_collection(c.name)
            count = collection.count()
            print(f"  Total chunks stored: {count}")

            # Peek at first few documents (default: 5)
            peek_limit = min(5, count)
            results = collection.peek(limit=peek_limit)
            docs = results.get("documents", [])
            if not docs:
                print("   No documents found.")
            else:
                for i, doc in enumerate(docs):
                    text = doc.replace("\n", " ")[:200]  # Show first 200 chars
                    print(f"   Chunk {i+1}: {text}...")
    except Exception as e:
        print(f"  ⚠️ Error inspecting {path}: {e}")

def main():
    print(f"🔍 Checking all Chroma collections under: {BASE_DIR}")

    if not os.path.exists(BASE_DIR):
        print(f"[!] Base directory not found: {BASE_DIR}")
        return

    # Inspect all immediate subdirectories
    subdirs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    if not subdirs:
        print("[!] No Chroma subdirectories found.")
        return

    for subdir in subdirs:
        inspect_collection(subdir)

    # Optional: check per-user session folders
    user_sessions_dir = os.path.join(BASE_DIR, "user_sessions")
    if os.path.exists(user_sessions_dir):
        print("\n📂 Checking user sessions:")
        user_dirs = [os.path.join(user_sessions_dir, u) for u in os.listdir(user_sessions_dir) if os.path.isdir(os.path.join(user_sessions_dir, u))]
        if not user_dirs:
            print("  No user session directories found.")
        else:
            for user_dir in user_dirs:
                inspect_collection(user_dir)
    else:
        print("\n[!] No user_sessions directory found.")

if __name__ == "__main__":
    main()




