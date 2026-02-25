from backend.db.session import engine
from backend.db.models import Base

Base.metadata.create_all(bind=engine)
print("Tables created successfully")