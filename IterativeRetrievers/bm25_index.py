import os
import sys
import subprocess

# Set up JAVA_HOME and JVM PATH in Python's environment object
java_home = r"C:\Program Files\Java\jdk-27"  # Replace with your actual JDK path
jvm_path = os.path.join(java_home, "bin", "server")

os.environ["JAVA_HOME"] = java_home
os.environ["PATH"] = jvm_path + os.pathsep + os.environ.get("PATH", "")

# command = [
#     sys.executable, "-m", "pyserini.index.lucene",
#     "--collection", "JsonCollection",
#     "--input", "../documents/corpora",
#     "--index", "../index/wiki_bm25", #2wiki_hotpot_musique_bm25
#     "--generator", "DefaultLuceneDocumentGenerator",
#     "--threads", "8",
#     "--storePositions",
#     "--storeDocvectors",
#     "--storeRaw"
# ]

# print("Starting Pyserini indexing...")

# # PASS os.environ TO THE SUBPROCESS HERE:
# subprocess.run(command, check=True, env=os.environ)

# print("Indexing completed successfully!")

from pyserini.search.lucene import LuceneSearcher
import time

searcher = LuceneSearcher("../index/wiki_bm25")
start = time.time()
hits = searcher.search("Which film came out earlier, Emmy Of Stork'S Nest or Indira Vizha?", k=5)
print("Query took",time.time()-start,"seconds")
print(f"Index check: Found {len(hits)} results.")
import json

for hit in hits:
    doc = searcher.doc(hit.docid)
    if doc:
        data = json.loads(doc.raw())
        print(f"ID: {data['id']}")
        print(f"Contents:\n{data['contents']}\n")