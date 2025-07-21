# 🎭 CineSphere

**Professional Movie Discovery Platform**

CineSphere is a sophisticated movie recommendation platform that combines machine learning algorithms with semantic search to deliver personalized movie suggestions. Built with modern web technologies and powered by a comprehensive dataset of **100 curated movies** across **10 major genres**.

## 🎬 Movie Dataset

Our comprehensive database features **100 carefully selected movies** spanning all major genres:

- **Action** (10 movies): The Dark Knight, Mad Max: Fury Road, John Wick, Die Hard, Terminator 2, The Matrix, Casino Royale, Mission Impossible, Speed, Raiders of the Lost Ark
- **Comedy** (10 movies): The Grand Budapest Hotel, Groundhog Day, Some Like It Hot, Dr. Strangelove, The Big Lebowski, Airplane!, Borat, Superbad, Anchorman, Coming to America  
- **Drama** (10 movies): The Shawshank Redemption, The Godfather, Schindler's List, 12 Angry Men, Forrest Gump, One Flew Over the Cuckoo's Nest, The Green Mile, Goodfellas, The Pianist, There Will Be Blood
- **Horror** (10 movies): The Exorcist, Halloween, The Shining, A Nightmare on Elm Street, Psycho, The Texas Chain Saw Massacre, Rosemary's Baby, The Babadook, Get Out, Hereditary
- **Science Fiction** (10 movies): 2001: A Space Odyssey, Blade Runner, Alien, Star Wars, E.T., Back to the Future, The Empire Strikes Back, Inception, Interstellar, The Matrix
- **Romance** (10 movies): Casablanca, The Princess Bride, Roman Holiday, Titanic, When Harry Met Sally, Sleepless in Seattle, Pretty Woman, Ghost, The Notebook, Eternal Sunshine
- **Thriller** (10 movies): Pulp Fiction, Se7en, The Silence of the Lambs, Zodiac, No Country for Old Men, Heat, Vertigo, North by Northwest, Rear Window, Gone Girl
- **Fantasy** (10 movies): The Lord of the Rings: Fellowship, Pan's Labyrinth, The Wizard of Oz, Big Fish, The Shape of Water, Edward Scissorhands, Harry Potter, Pirates of the Caribbean, Spirited Away
- **Animation** (10 movies): Toy Story, Finding Nemo, The Lion King, Up, WALL-E, Spirited Away, Inside Out, The Incredibles, Shrek, Spider-Man: Into the Spider-Verse
- **Mystery** (10 movies): The Maltese Falcon, Chinatown, The Third Man, L.A. Confidential, The Big Sleep, Murder on the Orient Express, The Usual Suspects, Memento, The Sixth Sense, Shutter Island

## 📂 Project Structure

```
CineSphere/
├── frontend/                 # React TypeScript Frontend
│   ├── src/
│   │   ├── components/      # UI Components
│   │   ├── pages/          # Application Pages
│   │   ├── services/       # API Integration
│   │   └── App.tsx         # Main App Component
│   └── package.json        # Frontend Dependencies
│
├── backend/                 # Python Backend Services
│   ├── SimpleRecommender.py    # ML Recommendation Engine
│   ├── Database.py             # Database Models & ORM
│   ├── Cache.py               # Caching System
│   ├── Embedding.py           # Text Embeddings
│   ├── VectorDB.py            # Vector Database
│   ├── LLM.py                 # Language Model Integration
│   ├── PromptTemplate.py      # AI Prompt Templates
│   └── Chunking.py            # Text Processing Utils
│
├── api/                     # FastAPI Web Server
│   └── main.py             # REST API Endpoints
│
├── data/                    # Data Storage
│   ├── movies.csv          # Comprehensive Movie Dataset (100 movies)
│   └── movieapp.db         # SQLite Database
│
├── create_comprehensive_dataset.py  # Dataset Generation Script
├── requirements.txt         # Python Dependencies
└── .env                    # Environment Variables
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **TF-IDF Vectorizer** - Content-based filtering
- **Cosine Similarity** - Movie similarity matching
- **SentenceTransformers** - Text embeddings (optional)
- **Scikit-learn** - Machine learning algorithms

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe development
- **Material-UI** - Component library
- **Framer Motion** - Animations
- **Axios** - HTTP client

### Database & Storage
- **SQLite** - Local database with 100 movies
- **File-based caching** - Performance optimization

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cinesphere
   ```

2. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   
   Create `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key  # Optional
   JWT_SECRET=your_jwt_secret_key
   ```

4. **Database Initialization**
   ```bash
   # The database comes pre-loaded with 100 movies
   # To regenerate the dataset:
   python create_comprehensive_dataset.py
   
   # Load into database
   python -c "
   from backend.Database import DatabaseManager
   db = DatabaseManager()
   db.load_movies_from_csv('data/movies.csv')
   print('Database initialized with 100 movies')
   "
   ```

5. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. **Start Backend API** (Terminal 1)
   ```bash
   # IMPORTANT: Always activate virtual environment first
   source .venv/bin/activate
   cd api
   python main.py
   ```
   API available at: `http://localhost:8000`

2. **Start Frontend** (Terminal 2)
   ```bash
   cd frontend
   npm start
   ```
   Web app available at: `http://localhost:3000`

## 🎯 Features

- **Smart Recommendations** - TF-IDF based content similarity across 100 movies
- **Genre Diversity** - 10 movies each across 10 major genres
- **Semantic Search** - Natural language movie discovery
- **Modern UI** - Responsive design with smooth animations
- **Real-time API** - Fast, cached responses
- **Professional Dataset** - Curated collection of acclaimed films

## 📋 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Sample Queries to Try:
- "horror movie with supernatural elements" → The Exorcist, The Shining, Psycho
- "john wick action movie" → John Wick, Die Hard, The Matrix
- "romantic comedy" → When Harry Met Sally, The Princess Bride
- "space adventure sci-fi" → Star Wars, 2001: A Space Odyssey, Interstellar
- "mystery crime thriller" → The Maltese Falcon, Chinatown, Se7en

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

---

**Built by [Nipun Kumar](https://www.linkedin.com/in/nipunkumar01/) and [Divyansh Sharma](https://www.linkedin.com/in/divyansh-sharma-b4793026b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)** 