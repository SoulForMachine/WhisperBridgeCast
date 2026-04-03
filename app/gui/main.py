import argparse
import logging
import sys

from app.gui.ui import CaptionerUI


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices="CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET".split(","),
        help="Logging level. Default: INFO",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s\t%(message)s")

    app = CaptionerUI()
    app.run_gui()


if __name__ == "__main__":
    main()
