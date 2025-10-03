import os
import json
import time
import random
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
import uuid

"""
Agent README (short):

Run the agent (mock mode):

    cd agent
    python3 data_collection_agent.py

Dependencies: requests, python-dotenv. For tests run pytest (if installed).
"""

"""Write a default configuration file with mock API settings."""
def _write_default_config(path):
    default = {
        "collection": {"mode": "mock", "base_delay": 0.5, "max_records": 5},
        "apis": {"rent": {"base_url": "https://api.example.com/rent", "fallback_url": "https://api.backup.example.com/rent", "city": "San Francisco", "limit": 5}}
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, "w") as f:
        json.dump(default, f, indent=2)

"""Agent for collecting and processing rental data, with mock and real API modes."""
class RentDataAgent:
    def __init__(self, config_file: str):
        """Initialize agent with config. Supports a mock mode for testing without external APIs.

        If `collection.mode` in the config is set to "mock", the agent will synthesize
        API responses and skip the API key requirement.
        """
        load_dotenv()
        self.config = self.load_config(config_file)
        self.mock_mode = self.config.get("collection", {}).get("mode") == "mock"

        # Only require an API key when not running in mock mode
        self.api_key = None
        if not self.mock_mode:
            self.api_key = os.getenv("RENT_API_KEY")
            if not self.api_key:
                raise RuntimeError("Missing RENT_API_KEY in your .env file and not running in mock mode")
        # Base directory for all outputs is the module folder (where this file lives)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.raw_data_dir = os.path.join(self.base_dir, "data", "raw")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)

        self.setup_logging()
        self.data_store = []
        self.collection_stats = {
            'start_time': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'apis_used': set(),
            'quality_scores': []
        }
        self.delay_multiplier = 1.0
        self.processing_log = []
        self.last_rate_headers = {}

    # ---------------- Configuration ----------------
    """Load configuration from JSON file, or write a default if missing/invalid."""
    def load_config(self, config_file):
        # Robustly load JSON config. If the file is missing or malformed,
        # write a default config next to the file and return that default.
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Config file {config_file} not found - writing default config.")
            _write_default_config(config_file)
            with open(config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Invalid JSON in config - overwrite with a safe default to allow mock runs
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Config file {config_file} contains invalid JSON - replacing with default.")
            _write_default_config(config_file)
            with open(config_file, "r") as f:
                return json.load(f)

    """Set up logging to file and console."""
    def setup_logging(self):
        log_file = os.path.join(self.logs_dir, "collection.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    # ---------------- Collection ----------------
    """Main execution loop for data collection, with error handling and final reporting."""
    def run_collection(self):
        self.logger.info("Starting Rent Data Collection Agent")
        try:
            while not self.collection_complete():
                data = self.collect_batch()
                if data:
                    self.process_and_store(data)

                self.assess_performance()
                self.respectful_delay()
        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
        finally:
            self.generate_final_report()
            self.save_raw_data()
            self.save_processed_data()


    """Check if the collection target (max_records) has been reached."""
    def collection_complete(self):
        max_records = self.config["collection"].get("max_records", 10)
        return len(self.data_store) >= max_records

    def collect_batch(self):
        """Make one API call per loop"""
        try:
            city = self.config.get("apis", {}).get("rent", {}).get("city", "Unknown")
        except Exception:
            city = "Unknown"
        return self.make_api_request(city)

    """Request rental data from the API (or generate mock data if in mock mode)."""
    def make_api_request(self, city: str):
        # Support a mock mode that synthesizes data for testing
        if self.mock_mode:
            self.collection_stats["total_requests"] += 1
            self.collection_stats["successful_requests"] += 1
            self.collection_stats["apis_used"].add("mock_rent_api")
            # synthetic listing
            listing = {
                "id": str(uuid.uuid4()),
                "city": city,
                "price": random.randint(500, 3500),
                "bedrooms": random.choice([1, 2, 3]),
                "bathrooms": random.choice([1, 1.5, 2]),
            }
            return {"listings": [listing]}

        base_url = self.config["apis"]["rent"]["base_url"]
        params = {
            "city": city,
            "limit": self.config["apis"]["rent"].get("limit", 10),
            "apikey": self.api_key
        }

        self.collection_stats["total_requests"] += 1
        self.collection_stats["apis_used"].add("rent_api")

        try:
            r = requests.get(base_url, params=params, timeout=20)
            # Capture rate limit headers for adaptive behaviour
            self.last_rate_headers = {k: v for k, v in r.headers.items()}
            if r.status_code == 429:
                self.logger.warning("Rate limit hit. Slowing down.")
                self.delay_multiplier *= 1.5
                self.collection_stats["failed_requests"] += 1
                return None
            r.raise_for_status()
            self.collection_stats["successful_requests"] += 1
            return r.json()
        except Exception as e:
            self.collection_stats["failed_requests"] += 1
            self.logger.error(f"Request failed for {city}: {e}")
            return None

    # ---------------- Data Processing ----------------
    """Process raw API response and store validated listings."""
    def process_and_store(self, raw):
        if "listings" not in raw:
            self.logger.warning("Unexpected API response")
            return
        for listing in raw["listings"]:
            processed = self.process_data(listing)
            if self.validate_data(processed):
                self.store_data(processed)

    """Normalize a single rental listing into a consistent record format."""
    def process_data(self, listing):
        rec = {
            "id": listing.get("id"),
            "city": listing.get("city"),
            "price": listing.get("price"),
            "bedrooms": listing.get("bedrooms"),
            "bathrooms": listing.get("bathrooms"),
            "timestamp": datetime.now().isoformat()
        }
        self.processing_log.append({"step": "normalize_rent", "time": datetime.now().isoformat()})
        return rec

    """Validate that a rental record contains required fields."""
    def validate_data(self, rec):
        required = ["id", "city", "price"]
        ok = all(rec.get(k) is not None for k in required)
        if not ok:
            self.logger.warning(f"Validation failed: {rec}")
        return ok

    """Store a processed record in the data store and log the result."""
    def store_data(self, rec):
        self.data_store.append(rec)
        self.logger.info(f"Stored listing {rec['id']} in {rec['city']}")
        # ✅ Print to terminal for screenshots
        print(f"Stored listing {rec['id']} in {rec['city']} at ${rec['price']}")

    # ---------------- Quality & Strategy ----------------
    """Calculate an overall quality score for collected data."""
    def assess_data_quality(self):
        if not self.data_store:
            return 0.0
        completeness = self.check_completeness()
        accuracy = self.check_accuracy()
        consistency = self.check_consistency()
        timeliness = self.check_timeliness()
        return (completeness + accuracy + consistency + timeliness) / 4

    # --------- Quality metric helpers ----------
    """Check what fraction of records have all required fields."""
    def check_completeness(self):
        # Fraction of records with all required fields
        required = ["id", "city", "price"]
        if not self.data_store:
            return 0.0
        ok = sum(1 for r in self.data_store if all(r.get(k) is not None for k in required))
        return ok / len(self.data_store)

    """Check if rental prices are valid and within expected bounds."""
    def check_accuracy(self):
        # Simple heuristic: price must be > 0 and within reasonable bounds
        if not self.data_store:
            return 0.0
        valid = sum(1 for r in self.data_store if isinstance(r.get("price"), (int, float)) and 0 < r.get("price") < 100000)
        return valid / len(self.data_store)

    """Verify that data types (like price) are consistent across records."""
    def check_consistency(self):
        # Check that types are consistent across records
        if not self.data_store:
            return 0.0
        types = set(type(r.get("price")) for r in self.data_store if r.get("price") is not None)
        return 1.0 if len(types) <= 1 else max(0.0, 1.0 - (len(types) - 1) * 0.5)

    """Check if records were collected recently (within 24 hours)."""
    def check_timeliness(self):
        # If data timestamps are recent relative to collection start, consider timely
        if not self.data_store:
            return 0.0
        now = datetime.now()
        recent = 0
        for r in self.data_store:
            ts = r.get("timestamp")
            try:
                dt = datetime.fromisoformat(ts)
                if (now - dt).total_seconds() < 3600 * 24:  # within 24h
                    recent += 1
            except Exception:
                continue
        return recent / len(self.data_store)

    """Return the fraction of successful API requests."""
    def get_success_rate(self):
        total = max(1, self.collection_stats["total_requests"])
        return self.collection_stats["successful_requests"] / total

    """Assess quality and adjust strategy if needed."""
    def assess_performance(self):
        q = self.assess_data_quality()
        self.collection_stats["quality_scores"].append(q)
        if self.get_success_rate() < 0.8:
            self.adjust_strategy()

    """Adapt strategy based on API success rate (slow down or speed up)."""
    def adjust_strategy(self):
        sr = self.get_success_rate()
        if sr < 0.5:
            self.delay_multiplier *= 2
            self.try_fallback_api()
            self.logger.info("Strategy: slowing down due to failures")
        elif sr > 0.9:
            self.delay_multiplier *= 0.8
            self.logger.info("Strategy: can speed up slightly")
        self.log_strategy_change()

    def try_fallback_api(self):
        """Attempt a single call to a fallback API from the config to verify availability."""
        fallback = self.config.get("apis", {}).get("rent", {}).get("fallback_url")
        if not fallback:
            return
        try:
            r = requests.get(fallback, timeout=10)
            if r.ok:
                self.logger.info("Fallback API reachable; will attempt to use it if primary fails")
                self.collection_stats["apis_used"].add("fallback")
        except Exception as e:
            self.logger.warning(f"Fallback API check failed: {e}")

    """Log when strategy changes due to performance issues."""
    def log_strategy_change(self):
        self.logger.info(f"Strategy change: delay_multiplier={self.delay_multiplier:.2f}, success_rate={self.get_success_rate():.2f}")

    def check_rate_limits(self):
        """Inspect last response headers for common rate-limit signals and adapt speed."""
        headers = self.last_rate_headers
        # Example: GitHub uses X-RateLimit-Remaining
        remaining = headers.get("X-RateLimit-Remaining") or headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                rem = int(remaining)
                if rem < 5:
                    self.logger.info("Approaching rate limit - increasing delay")
                    self.delay_multiplier *= 1.5
            except Exception:
                pass

    """Pause between API calls using delay + random jitter to respect limits."""
    def respectful_delay(self):
        base_delay = self.config["collection"].get("base_delay", 1.0)
        jitter = random.uniform(0.5, 1.5)
        time.sleep(base_delay * self.delay_multiplier * jitter)

    # ---------------- Reports ----------------
    """Generate summary, metadata, and quality reports at the end of collection."""
    def generate_final_report(self):
        summary = {
            "end_time": datetime.now().isoformat(),
            "total_records": len(self.data_store),
            "success_rate": round(self.get_success_rate(), 3),
            "apis_used": list(self.collection_stats["apis_used"]),
            "issues": "See logs/collection.log"
        }
        # JSON summary
        with open(os.path.join(self.reports_dir, "collection_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Human-readable markdown summary
        md_lines = [
            f"# Collection Summary\n",
            f"- end_time: {summary['end_time']}",
            f"- total_records: {summary['total_records']}",
            f"- success_rate: {summary['success_rate']}",
            f"- apis_used: {', '.join(summary['apis_used'])}",
            f"- issues: {summary['issues']}",
        ]
        with open(os.path.join(self.reports_dir, "collection_summary.md"), "w") as f:
            f.write("\n".join(md_lines))

        # Generate metadata and quality reports
        self.generate_metadata()
        self.generate_quality_report()

        self.logger.info("Final report generated: reports/collection_summary.json and reports/collection_summary.md")

    def save_raw_data(self):
        """Save collected data to data/raw/collected.json"""
        os.makedirs("data/raw", exist_ok=True)
        with open(os.path.join(self.raw_data_dir, "collected.json"), "w") as f:
            json.dump(self.data_store, f, indent=2)
        self.logger.info(f"Raw data saved to {os.path.join(self.raw_data_dir, 'collected.json')}")
        print(f"✅ Raw data saved to {os.path.join(self.raw_data_dir, 'collected.json')}")
        
    def save_processed_data(self):
        """Save validated/cleaned records to data/processed/processed.json"""
        processed_dir = os.path.join(self.base_dir, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        path = os.path.join(processed_dir, "processed.json")
        with open(path, "w") as f:
            json.dump(self.data_store, f, indent=2)

        self.logger.info(f"Processed data saved to {path}")
        print(f"✅ Processed data saved to {path}")


    # ---------------- Documentation & QA ----------------
    """Return a list of all APIs used during collection."""
    def get_sources_used(self):
        return list(self.collection_stats.get("apis_used", []))

    """Return the processing history log."""
    def get_processing_log(self):
        return self.processing_log

    """Infer variable types from the collected data."""
    def generate_data_dictionary(self):
        # Simple variables inference from first record
        if not self.data_store:
            return {}
        sample = self.data_store[0]
        dd = {k: type(v).__name__ for k, v in sample.items()}
        return dd

    """Return final calculated quality metrics for the dataset."""
    def calculate_final_quality_metrics(self):
        return {
            "overall_quality_score": round(self.assess_data_quality(), 3),
            "completeness": round(self.check_completeness(), 3),
            "accuracy": round(self.check_accuracy(), 3),
            "consistency": round(self.check_consistency(), 3),
            "timeliness": round(self.check_timeliness(), 3),
        }

    """Return the overall average quality score."""
    def get_overall_quality_score(self):
        return round(self.assess_data_quality(), 3)

    """Write dataset metadata including sources, variables, and quality metrics."""
    def generate_metadata(self):
        metadata = {
            "collection_info": {
                "collection_date": datetime.now().isoformat(),
                "agent_version": "1.0",
                "collector": "KeiraHanson",
                "total_records": len(self.data_store)
            },
            "data_sources": self.get_sources_used(),
            "quality_metrics": self.calculate_final_quality_metrics(),
            "processing_history": self.get_processing_log(),
            "variables": self.generate_data_dictionary()
        }
        with open(os.path.join(self.reports_dir, "dataset_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Dataset metadata written to {os.path.join(self.reports_dir, 'dataset_metadata.json')}")

    """Analyze completeness of collected data."""
    def analyze_completeness(self):
        return {"completeness": self.check_completeness(), "total_records": len(self.data_store)}

    """Summarize price distribution of rental listings."""
    def analyze_distribution(self):
        # simple price distribution
        prices = [r.get("price") for r in self.data_store if isinstance(r.get("price"), (int, float))]
        if not prices:
            return {}
        return {"min": min(prices), "max": max(prices), "mean": sum(prices) / len(prices)}

    """Detect outlier rental prices compared to median values."""
    def detect_anomalies(self):
        # detect prices that are 3x above median as anomalies (very simple)
        prices = sorted([r.get("price") for r in self.data_store if isinstance(r.get("price"), (int, float))])
        if not prices:
            return []
        mid = prices[len(prices) // 2]
        anomalies = [p for p in prices if p > mid * 3]
        return anomalies

    """Generate recommendations for improving data collection quality."""
    def generate_recommendations(self):
        qm = self.calculate_final_quality_metrics()
        recs = []
        if qm["completeness"] < 0.9:
            recs.append("Increase validation and reattempt missing fields")
        if qm["accuracy"] < 0.9:
            recs.append("Cross-check prices with alternative sources")
        return recs

    """Write a human-readable markdown quality report."""
    def create_readable_report(self, report, path="reports/quality_report.md"):
        lines = ["# Quality Report\n"]
        lines.append("## Summary")
        for k, v in report.get("summary", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Completeness Analysis")
        lines.append(json.dumps(report.get("completeness_analysis", {}), indent=2))
        lines.append("\n## Distribution")
        lines.append(json.dumps(report.get("data_distribution", {}), indent=2))
        lines.append("\n## Anomalies")
        lines.append(json.dumps(report.get("anomaly_detection", []), indent=2))
        lines.append("\n## Recommendations")
        lines.append("\n".join(report.get("recommendations", [])))
        with open(path, "w") as f:
            f.write("\n".join(lines))
        self.logger.info(f"Human-readable quality report written to {path}")

    """Generate detailed quality assessment report (JSON + Markdown)."""
    def generate_quality_report(self):
        report = {
            "summary": {
                "total_records": len(self.data_store),
                "collection_success_rate": self.get_success_rate(),
                "overall_quality_score": self.get_overall_quality_score()
            },
            "completeness_analysis": self.analyze_completeness(),
            "data_distribution": self.analyze_distribution(),
            "anomaly_detection": self.detect_anomalies(),
            "recommendations": self.generate_recommendations()
        }
        with open(os.path.join(self.reports_dir, "quality_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        self.create_readable_report(report, path=os.path.join(self.reports_dir, "quality_report.md"))
        self.logger.info(f"Quality report generated: {os.path.join(self.reports_dir, 'quality_report.json')}")

# ---------------- Script Entry ----------------


if __name__ == "__main__":
    # Place config next to this file so relative paths are predictable
    module_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(module_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"No config found at {cfg_path}, writing default mock config.")
        _write_default_config(cfg_path)

    agent = RentDataAgent(cfg_path)
    agent.run_collection()


def _self_test():
    """Very small self-test to check core flows (run: python data_collection_agent.py --self-test)"""
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    cfg_path = os.path.join(tmp.name, "config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "collection": {"mode": "mock", "base_delay": 0.01, "max_records": 1},
            "apis": {"rent": {"city": "TestCity", "limit": 1}}
        }, f)
    a = RentDataAgent(cfg_path)
    # test process_data and validate
    sample = {"id": "1", "city": "Test", "price": 100}
    p = a.process_data(sample)
    assert a.validate_data(p)
    # test collection loop (one cycle)
    a.config["collection"]["max_records"] = 1
    a.run_collection()
    assert len(a.data_store) >= 1

if __name__ == "__main__" and "--self-test" in os.sys.argv:
    _self_test()


def run_pytest_inline():
    """Write minimal pytest test file to a temp dir and run pytest to validate core functions."""
    import tempfile, subprocess, textwrap

    testsrc = textwrap.dedent('''
    import json
    from data_collection_agent import RentDataAgent

    def test_process_and_validate_basic(tmp_path):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "collection": {"mode": "mock", "base_delay": 0.01, "max_records": 1},
            "apis": {"rent": {"city": "TestCity", "limit": 1}}
        }))
        agent = RentDataAgent(str(cfg))
        sample = {"id": "1", "city": "TestCity", "price": 100}
        processed = agent.process_data(sample)
        assert agent.validate_data(processed)

    def test_assess_quality_empty(tmp_path):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "collection": {"mode": "mock", "base_delay": 0.01, "max_records": 1},
            "apis": {"rent": {"city": "TestCity", "limit": 1}}
        }))
        agent = RentDataAgent(str(cfg))
        assert agent.assess_data_quality() == 0.0
    ''')

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test_agent.py")
        with open(p, "w") as f:
            f.write(testsrc)
        try:
            res = subprocess.run(["pytest", p, "-q"], capture_output=True, text=True, check=False)
            print(res.stdout)
            if res.returncode != 0:
                print(res.stderr)
            return res.returncode
        except FileNotFoundError:
            print("pytest not installed in this environment; install with pip install pytest")
            return 2


