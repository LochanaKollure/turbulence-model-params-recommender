"""
Main entry point for the Turbulence Model Parameter Recommender application.
"""

import os
import sys
import subprocess
import logging

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease copy .env.template to .env and add your API keys.")
        return False
    
    print("Environment variables configured")
    return True

def run_streamlit_app():
    """Launch the Streamlit application."""
    try:
        # Change to the UI directory
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'streamlit_app.py')
        
        if not os.path.exists(ui_path):
            print(f"Streamlit app not found at: {ui_path}")
            return False
        
        print("Starting Turbulence Model Parameter Recommender...")
        print("Access the application at: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        return False
    
    return True

def main():
    """Main entry point."""
    print("Turbulence Model Parameter Recommender")
    print("=" * 45)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run the application
    if not run_streamlit_app():
        sys.exit(1)

if __name__ == "__main__":
    main()