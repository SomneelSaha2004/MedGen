#!/usr/bin/env python
"""
MedGen - AI-Powered Synthetic Medical Data Generation Platform

This module serves as the main entry point for the MedGen application.
It can be used to run individual components or the full application.

Usage:
    python main.py                    # Start the backend server
    python main.py --eval             # Run evaluation pipeline
    python main.py --privacy          # Run privacy assessment
    python main.py --multi-eval       # Run multi-dataset evaluation
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point for MedGen."""
    parser = argparse.ArgumentParser(
        description="MedGen - AI-Powered Synthetic Medical Data Generation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start the backend API server
  python main.py --eval             Run the evaluation pipeline
  python main.py --privacy          Run privacy risk assessment
  python main.py --multi-eval       Run multi-dataset evaluation
  python main.py --scale            Run scaling experiments
        """
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the ML evaluation pipeline"
    )
    parser.add_argument(
        "--privacy",
        action="store_true",
        help="Run privacy risk assessment"
    )
    parser.add_argument(
        "--multi-eval",
        action="store_true",
        help="Run multi-dataset evaluation"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Run scaling experiments"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for the backend server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the backend server (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Run the appropriate component
    if args.eval:
        print("🔬 Running evaluation pipeline...")
        from basic_eval_pipeline import run_evaluation_pipeline
        run_evaluation_pipeline(
            dataset_path="./evals/dataset/pima-diabetes.csv",
            output_dir="./results"
        )
    elif args.privacy:
        print("🔒 Running privacy assessment...")
        from anonymeter_privacy_eval import main as privacy_main
        privacy_main()
    elif args.multi_eval:
        print("🔬 Running multi-dataset evaluation...")
        from multi_dataset_pipeline import main as multi_main
        multi_main()
    elif args.scale:
        print("📊 Running scaling experiments...")
        import scaling_pipeline
    else:
        # Default: start the backend server
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                    🏥 MedGen Backend                          ║
║     AI-Powered Synthetic Medical Data Generation Platform     ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        from backend import app
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

