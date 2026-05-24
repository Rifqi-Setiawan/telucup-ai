from sqlalchemy import create_engine, Column, Integer, String, Float, Enum, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os

load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Mapping tabel face_embeddings (Ground Truth / Reference vectors)
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey('players.id'), unique=True, index=True)
    embedding = Column(Vector(512))

# Mapping tabel photo_faces (AI detection results)
class PhotoFace(Base):
    __tablename__ = "photo_faces"

    id = Column(Integer, primary_key=True, index=True)
    event_photo_id = Column(Integer, index=True)
    matched_player_id = Column(Integer, nullable=True)
    validation_status = Column(String, default="pending")
    similarity_score = Column(Float, nullable=True)
    bounding_box = Column(JSON, nullable=True)
    face_encoding = Column(Vector(512)) # pgvector 512 dimensi (output AdaFace)
    
    # Abaikan timestamps untuk insert cepat, atau tambahkan jika wajib