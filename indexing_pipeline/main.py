import os
import sys
import time
import logging

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "indexing-pipeline")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RUN_ID = os.getenv("RUN_ID", str(int(time.time())))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(PIPELINE_NAME)


def _exit_hard(stage: str, exc: Exception):
    logger.error(f"pipeline_stage_failed stage={stage} error={str(exc)} run_id={RUN_ID}")
    sys.exit(1)


def run_extract_load():
    logger.info(f"stage_start extract_load run_id={RUN_ID}")
    try:
        from ELT.extract_load.web_scraper import run as extract_run
        extract_run()
        logger.info(f"stage_complete extract_load run_id={RUN_ID}")
    except Exception as e:
        _exit_hard("extract_load", e)


def run_parse_chunk_store():
    logger.info(f"stage_start parse_chunk_store run_id={RUN_ID}")
    try:
        from ELT.parse_chunk_store.router import run as parse_run
        parse_run()
        logger.info(f"stage_complete parse_chunk_store run_id={RUN_ID}")
    except Exception as e:
        _exit_hard("parse_chunk_store", e)


def run_embed_and_index():
    logger.info(f"stage_start embed_and_index run_id={RUN_ID}")
    try:
        from embed_and_index import run as embed_run
        embed_run()
        logger.info(f"stage_complete embed_and_index run_id={RUN_ID}")
    except Exception as e:
        _exit_hard("embed_and_index", e)


def main():
    logger.info(f"pipeline_start name={PIPELINE_NAME} run_id={RUN_ID}")
    run_extract_load()
    run_parse_chunk_store()
    run_embed_and_index()
    logger.info(f"pipeline_complete name={PIPELINE_NAME} run_id={RUN_ID}")


if __name__ == "__main__":
    main()
