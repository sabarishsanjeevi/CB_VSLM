#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple_interface.py - Simple interface for Context-Specialized Small Language Models

This script provides a user-friendly interface for interacting with models trained
with the cs_slm.py implementation.
"""

import os
import sys
import json
import logging
import argparse
import torch
from pathlib import Path

# Import necessary classes from cs_slm
try:
    from cs_slm import ModelInterface
except ImportError as e:
    print(f"Error importing from cs_slm.py: {e}")
    print("Make sure cs_slm.py is in the current directory or Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the interface"""
    parser = argparse.ArgumentParser(
        description='Interface for Context-Specialized Small Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model path
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model directory')

    # Response generation settings
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated responses')

    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode with command line interface')

    parser.add_argument('--prompt', type=str,
                        help='Single prompt to process (non-interactive mode)')

    return parser.parse_args()


def run_interactive_mode(interface):
    """
    Run the interface in interactive command-line mode

    Args:
        interface: Initialized ModelInterface object
    """
    print("\n" + "=" * 50)
    print("Context-Specialized Small Language Model Interface")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50 + "\n")

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\nUser: ")

            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break

            # Generate and show response
            response = interface.generate_response(user_input)
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def process_single_prompt(interface, prompt):
    """
    Process a single prompt

    Args:
        interface: Initialized ModelInterface object
        prompt: Text prompt to process
    """
    try:
        # Generate response
        response = interface.generate_response(prompt)
        print(f"User: {prompt}")
        print(f"Assistant: {response}")
    except Exception as e:
        print(f"Error processing prompt: {e}")


def main():
    """Main function"""
    args = parse_arguments()

    try:
        # Load the model
        logger.info(f"Loading model from {args.model_path}")
        interface = ModelInterface(args.model_path, max_length=args.max_length)
        logger.info("Model loaded successfully")

        # Determine mode
        if args.interactive:
            # Interactive mode
            run_interactive_mode(interface)
        elif args.prompt:
            # Single prompt mode
            process_single_prompt(interface, args.prompt)
        else:
            # Default to interactive if no prompt provided
            print("No prompt provided, starting interactive mode")
            run_interactive_mode(interface)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
