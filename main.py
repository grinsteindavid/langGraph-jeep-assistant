"""Main application for Jeep Patriot diagnostic assistant."""

import os
# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from dotenv import load_dotenv
from patriot_agent import PatriotAgent

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to run the Jeep Patriot diagnostic assistant."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize the agent
    pdf_path = "2011-patriot manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Manual PDF not found at {pdf_path}")
        return
    
    print("🚗 Initializing Jeep Patriot Diagnostic Assistant...")
    print("📖 Loading manual content...")
    
    try:
        agent = PatriotAgent(pdf_path)
        print("✅ Assistant ready!")
        print("\n" + "="*60)
        print("JEEP PATRIOT DIAGNOSTIC ASSISTANT")
        print("="*60)
        print("Ask me about any issues with your 2011 Jeep Patriot.")
        print("I'll consult the official manual to help diagnose problems.")
        print("Type 'quit' or 'exit' to end the session.")
        print("="*60 + "\n")
        
        while True:
            user_input = input("🔧 Describe your Patriot issue: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Jeep Patriot Diagnostic Assistant!")
                break
            
            if not user_input:
                continue
            
            print("\n🔍 Analyzing issue and consulting manual...")
            
            try:
                diagnosis = agent.diagnose(user_input)
                print("\n📋 DIAGNOSTIC RESPONSE:")
                print("-" * 40)
                print(diagnosis)
                print("-" * 40 + "\n")
                
            except Exception as e:
                print(f"❌ Error during diagnosis: {e}")
                print("Please try rephrasing your question.\n")
    
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}")

if __name__ == "__main__":
    main()
