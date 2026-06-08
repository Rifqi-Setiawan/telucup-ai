import sys
import os
sys.path.append('.')

from database import SessionLocal, FaceEmbedding

def test_db():
    print("Testing DB Connection...")
    try:
        db = SessionLocal()
        total_embeddings = db.query(FaceEmbedding).count()
        print(f"[SUCCESS] Connected to MySQL. Total FaceEmbeddings in DB: {total_embeddings}")
        
        # Test pulling one and checking JSON parsing
        first = db.query(FaceEmbedding).first()
        if first:
            print(f"Sample Embedding Type: {type(first.embedding)}")
            if isinstance(first.embedding, str):
                import json
                try:
                    parsed = json.loads(first.embedding)
                    print(f"String JSON parsed! Length: {len(parsed)}")
                except:
                    print("Could not parse JSON string.")
            elif isinstance(first.embedding, list):
                print(f"Native JSON List! Length: {len(first.embedding)}")
                
    except Exception as e:
        print(f"[FAILED] to connect or query: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    test_db()
