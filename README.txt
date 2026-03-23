RUN COMMANDS 
 Option 1 — Docker Compose (recommended, includes Redis)                                                                                                                                 
                                                                                  
  cp .env.example .env          # fill in your API keys                                                                                                                                   
  docker compose up -d          # backend :8000, redis :6379                      
  docker compose logs -f        # watch logs

  Option 2 — Docker only (no Redis, in-memory fallback)

  cp .env.example .env
  docker build -t trinx-backend .
  docker run -p 8000:8000 --env-file .env trinx-backend

  Option 3 — Local Python (no Docker)

  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  playwright install chromium    # needed for web scraping
  cp .env.example .env           # fill in keys
  python main.py                 # runs on :8000

  Verify it's running

  curl http://localhost:8000/health
  curl "http://localhost:8000/debug/test-intent?query=hello"
  curl -X POST "http://localhost:8000/debug/test-scrape?url=https://example.com"

  Stop

  docker compose down            # compose
  # or ctrl+c                    # local python

