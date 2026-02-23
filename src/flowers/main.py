import uvicorn
import os
import sys
from pathlib import Path

# Add 'src' directory to sys.path so 'flowers' package can be imported
# Path(__file__).parent is src/flowers/
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Use "0.0.0.0" to listen on all interfaces, or "127.0.0.1" for local only
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting server on {host}:{port}")
    # We use the string "api:app" to support the 'reload' feature
    # The string refers to the package 'flowers.api' and the variable 'app'
    uvicorn.run("flowers.api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    main()
