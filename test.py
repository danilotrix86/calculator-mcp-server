import os
import sys
import argparse
import requests
from dotenv import load_dotenv


def load_environment() -> None:
    # Load from project root .env (if present) and app/.env (your key is there now)
    try:
        load_dotenv()
    except Exception:
        pass
    try:
        load_dotenv(os.path.join("app", ".env"))
    except Exception:
        pass


def build_headers() -> dict:
    # Do NOT send API key from client; server should load it from .env
    return {"Content-Type": "application/json"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the /api/solve endpoint")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server",
    )
    parser.add_argument(
        "--text",
        default="differentiate x + 3x and provide the final simplified result",
        help="Math query text to send",
    )
    args = parser.parse_args()

    load_environment()

    url = f"{args.base_url.rstrip('/')}/api/solve"
    payload = {"text": args.text}
    headers = build_headers()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not response.ok:
        print(response.text)
        sys.exit(response.status_code)

    data = response.json()
    print("used_tool:", data.get("used_tool"))
    print("tool_called:", data.get("tool_called"))
    print("answer:", data.get("answer"))
    if not data.get("answer"):
        print(
            "No answer text returned. Try a more explicit prompt, e.g. "
            "'differentiate x^2 + 3x and state the result'."
        )


if __name__ == "__main__":
    main()


