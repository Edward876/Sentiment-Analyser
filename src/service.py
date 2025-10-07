"""Service layer: CLI and optional HTTP server.

CLI: python -m src.service "text" [--score] [--model PATH]
HTTP: ENABLE_HTTP=1 python -m src.service
"""

import argparse
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.strategy import analyze_strategy, analyze_strategy_score


def run_cli(text: str, score_mode: bool = False, model_path: str = "models/sentiment.joblib") -> None:
    """Run CLI analysis and print result.
    
    Args:
        text: Input text
        score_mode: If True, print numeric score; else print label
        model_path: Path to model file
    """
    if score_mode:
        result = analyze_strategy_score(text, model_path=model_path)
        print(f"{result:.3f}")
    else:
        result = analyze_strategy(text, model_path=model_path)
        print(result)


class SentimentHandler(BaseHTTPRequestHandler):
    """HTTP request handler for sentiment analysis.
    
    GET /analyze?text=...&score=0|1
    """
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        
        if parsed.path != "/analyze":
            self.send_error(404, "Not Found")
            return
        
        params = parse_qs(parsed.query)
        
        if "text" not in params:
            self.send_error(400, "Missing 'text' parameter")
            return
        
        text = params["text"][0]
        score_mode = params.get("score", ["0"])[0] == "1"
        
        # Analyze
        try:
            if score_mode:
                result = analyze_strategy_score(text)
                response = f"{result:.3f}"
            else:
                result = analyze_strategy(text)
                response = result
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        
        except Exception as e:
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass


def run_http(port: int = 8000) -> None:
    """Start HTTP server.
    
    Args:
        port: Port to listen on
    """
    server = HTTPServer(("0.0.0.0", port), SentimentHandler)
    print(f"HTTP server running on http://0.0.0.0:{port}")
    print(f"Try: curl 'http://localhost:{port}/analyze?text=I%20love%20pizza'")
    server.serve_forever()


def main():
    """Main entry point."""
    # Check if HTTP mode
    if os.environ.get("ENABLE_HTTP") == "1":
        port = int(os.environ.get("PORT", "8000"))
        run_http(port)
        return
    
    # CLI mode
    parser = argparse.ArgumentParser(
        description="Sentiment analyzer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python -m src.service \"I love pizza!\""
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (required unless --test)",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Print numeric score instead of label",
    )
    parser.add_argument(
        "--model",
        default="models/sentiment.joblib",
        help="Path to model file (default: models/sentiment.joblib)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pytest and exit",
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run pytest
        result = subprocess.run(["pytest", "-q"], cwd=".")
        sys.exit(result.returncode)
    
    if not args.text:
        parser.print_help()
        sys.exit(2)
    
    run_cli(args.text, score_mode=args.score, model_path=args.model)


if __name__ == "__main__":
    main()
