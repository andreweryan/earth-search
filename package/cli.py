import argparse
from .main import function


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arg",
        nargs="*",
        required=True,
        help="",
    )

    args = parser.parse_args()

    function(args.arg)


if __name__ == "__main__":
    main()
