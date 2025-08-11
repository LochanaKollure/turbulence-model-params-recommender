"""
Simple validation script to check project structure without external dependencies.
"""

import os
import sys

def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        'requirements.txt',
        '.env.template',
        '.gitignore',
        'README.md',
        'src/__init__.py',
        'src/config.py',
        'src/turbulence_models.py',
        'src/document_processor.py',
        'src/vector_store.py',
        'src/retrieval_system.py',
        'src/parameter_generator.py',
        'src/rag_pipeline.py',
        'ui/streamlit_app.py',
        'scripts/populate_database.py',
        'tests/test_basic.py',
        'run_app.py'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print("Project Structure Validation")
    print("=" * 35)
    
    print(f"Existing files: {len(existing_files)}")
    for file_path in existing_files:
        print(f"   + {file_path}")
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print(f"\nAll {len(required_files)} required files are present!")
    return True

def check_directory_structure():
    """Check directory structure."""
    required_dirs = ['src', 'ui', 'scripts', 'tests', 'data']
    
    print("\nDirectory Structure")
    print("=" * 25)
    
    for dir_name in required_dirs:
        dir_path = os.path.join(os.getcwd(), dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"   + {dir_name}/")
        else:
            print(f"   - {dir_name}/ (missing)")
    
    return True

def check_requirements_file():
    """Check requirements.txt content."""
    req_file = os.path.join(os.getcwd(), 'requirements.txt')
    
    if not os.path.exists(req_file):
        print("\nrequirements.txt not found")
        return False
    
    print("\nRequirements File")
    print("=" * 20)
    
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    required_packages = [
        'pinecone-client',
        'openai', 
        'streamlit',
        'requests',
        'PyPDF2',
        'sentence-transformers',
        'python-dotenv',
        'pydantic'
    ]
    
    found_packages = []
    content = ''.join(lines).lower()
    
    for package in required_packages:
        if package.lower() in content:
            found_packages.append(package)
            print(f"   + {package}")
        else:
            print(f"   - {package} (missing)")
    
    print(f"\nFound {len(found_packages)}/{len(required_packages)} required packages")
    
    return len(found_packages) >= len(required_packages) * 0.8  # 80% threshold

def main():
    """Run validation checks."""
    print("Turbulence Model Parameter Recommender - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Directory Structure", check_directory_structure), 
        ("Requirements", check_requirements_file)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\nError in {check_name} check: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("VALIDATION PASSED - Project structure looks good!")
        print("\nNext Steps:")
        print("1. Copy .env.template to .env and add your API keys")
        print("2. Create and activate virtual environment:")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate  (Windows)")
        print("   source venv/bin/activate  (Linux/Mac)")
        print("3. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("4. Populate database:")
        print("   python scripts/populate_database.py --mode populate")
        print("5. Run application:")
        print("   python run_app.py")
    else:
        print("VALIDATION FAILED - Please fix the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)