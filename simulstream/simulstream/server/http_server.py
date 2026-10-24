# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import logging
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import simulstream


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.http_server')


class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, config=None, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/config.yaml":
            # Load template
            with open(self.config) as f:
                config = f.read()

            # Send response
            self.send_response(200)
            self.send_header("Content-type", "text/yaml; charset=utf-8")
            self.end_headers()
            self.wfile.write(config.encode("utf-8"))
        else:
            super().do_GET()


def cli_main():
    """
    Simulstream HTTP server command-line interface (CLI) entry point.

    This function parses command-line arguments and starts the asynchronous :func:`main` routine.

    Example usage::

        $ python http_server.py --config config/server.yaml --directory webdemo

    Command-line arguments:

    - ``bind`` (str): IP/address on which to serve [default: 127.0.0.1].
    - ``port`` (int): Port on which to serve [default: 8000].
    - ``--config`` (str): Path to the server configuration file.
    - ``--directory`` (str): Path to the server configuration file.

    .. note::
        The server currently does not support secure connection through HTTPS
    """
    LOGGER.info(f"HTTP server version: {simulstream.__version__}")
    parser = argparse.ArgumentParser(description="Simulstream http.server")
    parser.add_argument(
        "--bind", "-b", default="127.0.0.1",
        help="Specify alternate bind address [default: 127.0.0.1]")
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="Specify alternate port [default: 8000]")
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to configuration file (YAML)")
    parser.add_argument(
        "--directory", "-d", default="./webdemo",
        help="Path to the directory containing the HTML web demo [default: ./webdemo]")
    args = parser.parse_args()

    custom_handler = partial(CustomHandler, config=args.config, directory=args.directory)
    httpd = ThreadingHTTPServer((args.bind, args.port), custom_handler)
    LOGGER.info(f"Serving directory {args.directory}")
    LOGGER.info(f"Serving on http://{args.bind}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    cli_main()
