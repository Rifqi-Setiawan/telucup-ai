from sqlalchemy import create_engine, Column, Integer, String, Float, Enum, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# Sesuaikan dengan kredensial PostgreSQL lokal Anda
# Format: postgresql://username:password@host:port/nama_database
SQLALCHEMY_DATABASE_URL = "postgresql://root:pamelo04123@127.0.0.1:5432/telucup"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Kita hanya perlu me-mapping tabel yang akan diubah oleh AI (photo_faces)
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