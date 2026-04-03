import argparse
import logging
import multiprocessing as mp
import os
import time

from app.server.tcp_server import WhisperServer


def main(argv=None):
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host adress. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="Port number. Default: 5000")
    parser.add_argument("--write-wav", action="store_true", help="Write received audio to a wav file.")
    parser.add_argument("--write-transcript", action="store_true", help="Write received transcript to a text file.")
    parser.add_argument("--warmup-file", type=str, default="data/samples_jfk.wav", help="Provide the audio file used to warm up the whisper model.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices="CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET".split(","),
        help="Logging level. Default: INFO",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s\t%(message)s")

    srv_args = vars(args)
    srv_args.pop("log_level")  # log level is used to configure the root logger, no need to pass it to the server.

    whisper_server = WhisperServer(**srv_args)
    whisper_server.start()

    logger = logging.getLogger(__name__)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping server...")
    finally:
        whisper_server.stop()
        os._exit(0)


if __name__ == "__main__":
    main()
