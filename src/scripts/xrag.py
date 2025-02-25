import sys

from src.classes.utils.DebugLogger import DebugLogger
from src.classes.utils.EnvLoader import EnvLoader
from src.classes.xrag.ContractAnalyzer import ContractAnalyzer

EnvLoader(env_dir="src/config").load_env_files()


def main():
    """
    Main script to initialize and run the ContractAnalyzer.
    """
    logger = DebugLogger()

    try:
        # Initialize ContractAnalyzer with dataset path and mode
        analyzer = ContractAnalyzer("dataset/manually-verified-{}", mode="both", use_multiprocessing=True)

        # Start contract analysis
        logger.info("Starting contract analysis...")
        analyzer.analyze_contracts()
        logger.info("Contract analysis completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during contract analysis: {e}")
        sys.exit(1)  # Exit with a non-zero code on failure


if __name__ == "__main__":
    main()
