def parse_testcases_from_xml(xml_path):
    """Parse testcases from JUnit XML, inferring status directly from tags."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    tcs = []
    for tc in tree.findall('.//testcase'):
        name = tc.get('classname', '') + '.' + tc.get('name', '')
        # Status logic: failed > skipped > passed
        status = 'passed'
        if tc.find('failure') is not None:
            status = 'failed'
        elif tc.find('skipped') is not None:
            status = 'skipped'
        # Parse properties
        props = {}
        props_node = tc.find('properties')
        if props_node is not None:
            for p in props_node.findall('property'):
                props[p.get('name')] = p.get('value')
        tcs.append({
            "name": name,
            "status": status,
            "properties": props
        })
    return tcs
import json
import os
import sys
import re
import threading
import time
import webbrowser
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources
import shutil
from datetime import datetime

# All output files will be placed under the "dashboard" directory

def main():
    # Load environment
    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Paths
    XML_PATH = './pytest-results.xml'
    LOG_PATH = './automation.log'
    dashboard_dir = Path.cwd() / "dashboard"
    results_json_path = dashboard_dir / "results.json"
    MODEL = 'gpt-4o-mini'

    # Read inputs
    xml = Path(XML_PATH).read_text(encoding='utf-8')
    log = Path(LOG_PATH).read_text(encoding='utf-8') if Path(LOG_PATH).exists() else ''

    def print_dots(stop_event):
        while not stop_event.is_set():
            print('.', end='', flush=True)
            time.sleep(0.6)
        print()

    print("Generating RCA report via OpenAI... (may take up to a minute)")
    stop_event = threading.Event()
    t = threading.Thread(target=print_dots, args=(stop_event,))
    t.start()

    # Build prompts
    system_p = (
        "You are an expert QA automation and failure-analysis assistant. "
        "Given pytest JUnit XML results (which may include <properties> tags, e.g. "
        "<property name=\"negative\" value=\"true\"/>) and an optional log file, "
        "extract and summarize the following in JSON:\n"
        "1. summary: passed, failed, skipped counts, and trends.\n"
        "2. anomalies: recurring errors or warnings with counts.\n"
        "3. root_cause: modules/scripts with failure counts.\n"
        "4. recommendations: actionable steps.\n"
        "Also include a 'testcases' array, where each testcase has:\n"
        "   - name\n"
        "   - status\n"
        "   - properties: object with any custom properties (e.g. negative: true)\n"
        "When summarizing anomalies, root causes, and recommendations, "
        "ignore anomalies or failures that occur exclusively in tests marked as negative=true. "
        "Do not recommend fixing issues that only appear in negative tests.\n"
        "Add a section 'failure_classification', with two lists:\n"
        "- real_bugs: list of testcases or issues likely to be actual bugs in the system under test\n"
        "- test_issues: list of testcases or issues likely due to problems in the test code, test data, or test environment.\n"
        "Classify each failure or anomaly to the most appropriate list, with a brief explanation."
    )

    user_p = f"<!-- XML_BEGIN -->\n{xml}\n<!-- XML_END -->\n<!-- LOG_BEGIN -->\n{log}\n<!-- LOG_END -->"

    # Call LLM
    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system_p},
                  {"role":"user","content":user_p}],
        temperature=0
    )
    stop_event.set()
    t.join()
    raw = resp.choices[0].message.content or ""
    match = re.search(r'(\{[\s\S]*\})', raw)
    if not match:
        print("ERROR: could not extract JSON", file=sys.stderr)
        sys.exit(1)
    data = json.loads(match.group(1))

    # Always use testcases parsed directly from the XML, NOT from LLM!
    testcases = parse_testcases_from_xml(XML_PATH)
    data['testcases'] = testcases

    # --- Improved negative test handling and reporting ---
    # Count negative and regular test results separately based on parsed XML testcases.
    neg_passed = neg_failed = neg_skipped = 0
    reg_passed = reg_failed = reg_skipped = 0

    for tc in testcases:
        is_negative = tc['properties'].get('negative') in [True, "true", "True", 1, "1"]
        status = tc['status']
        if is_negative:
            if status == 'passed':
                neg_passed += 1
            elif status == 'failed':
                neg_failed += 1
            elif status == 'skipped':
                neg_skipped += 1
        else:
            if status == 'passed':
                reg_passed += 1
            elif status == 'failed':
                reg_failed += 1
            elif status == 'skipped':
                reg_skipped += 1

    data['summary'] = {
        "passed": reg_passed + neg_passed,
        "failed": reg_failed + neg_failed,
        "skipped": reg_skipped + neg_skipped,
        "regular_passed": reg_passed,
        "regular_failed": reg_failed,
        "regular_skipped": reg_skipped,
        "negative_total": neg_passed + neg_failed + neg_skipped,
        "negative_passed": neg_passed,
        "negative_failed": neg_failed,
        "negative_skipped": neg_skipped,
        "explanation": "Negative tests (with property negative=true) are counted and displayed separately for full transparency."
    }

    # Extract execution times and test names
    times = []
    name_time = []
    try:
        tree = ET.parse(XML_PATH)
        for tc in tree.findall('.//testcase'):
            t = tc.get('time')
            name = tc.get('classname') + '.' + tc.get('name')
            if t:
                ft = float(t)
                times.append(ft)
                name_time.append((name, ft))
    except Exception as e:
        print(f"Warning: couldn't parse execution times: {e}", file=sys.stderr)

    data['execution_times'] = times

    # Identify slowest tests (top durations)
    name_time.sort(key=lambda x: x[1], reverse=True)
    # take all tests in highest bin
    if times:
        counts, bins = np.histogram(times, bins=10)
        slow_threshold = bins[-2]  # lower edge of highest bin
        slow_tests = [n for n, t in name_time if t >= slow_threshold]
    else:
        slow_tests = []

    data['slowest_tests'] = slow_tests

    # Generate histogram chart if times exist
    if times:
        counts, bins = np.histogram(times, bins=10)
        fig, ax = plt.subplots()
        ax.bar(range(len(counts)), counts)
        ax.set_xticks(range(len(bins)-1))
        ax.set_xticklabels([f"{bins[i]:.1f}-{bins[i+1]:.1f}s" for i in range(len(bins)-1)], rotation=45, ha='right')
        ax.set_title('Execution Time Distribution')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        data['chart_time_dist'] = f"data:image/png;base64,{img_b64}"
        plt.close(fig)

    # --- New order for output and assets ---

    # Copy (and update) dashboard assets, but do NOT delete dashboard_dir or its contents (history, etc.)
    try:
        assets_path = importlib.resources.files("rca_report").joinpath("assets")
        if not dashboard_dir.exists():
            dashboard_dir.mkdir(parents=True)
        for item in assets_path.iterdir():
            dest = dashboard_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"\n📊 Dashboard assets copied to: {dashboard_dir}")

        print(f"Open your report in your browser:\nfile://{dashboard_dir / 'index.html'}\n")
        # Commented out automatic browser opening to comply with instructions
        # try:
        #     webbrowser.open(f"file://{dashboard_dir / 'index.html'}")
        # except Exception as e:
        #     print(f"Could not open browser automatically: {e}")
        # Print user guidance for local web server
        print(
            "⚠ NOTE: If you see an empty dashboard or a CORS error in your browser, it's due to browser security restrictions when using file://.\n\n"
            "To view the dashboard with full data, run:\n"
            "    python -m http.server 8000 --directory dashboard\n\n"
            "Then open:\n"
            "    http://localhost:8000/index.html\n"
        )
    except Exception as e:
        print(f"Could not copy dashboard assets: {e}")

    # Ensure dashboard/data/history directory exists before writing
    history_dir = dashboard_dir / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    history_file = history_dir / f"results_{timestamp}.json"
    Path(history_file).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote history JSON to {history_file}")

    # Copy newest history file to dashboard/results.json
    try:
        shutil.copy2(history_file, results_json_path)
        print(f"Copied newest results to {results_json_path}")
    except Exception as e:
        print(f"ERROR: could not copy newest results to {results_json_path}: {e}", file=sys.stderr)

    # Print summary of available history JSON files sorted by creation date
    try:
        history_files = list(history_dir.glob("results_*.json"))
        history_files.sort(key=lambda f: f.stat().st_ctime)
        print("\nAvailable history reports:")
        for hf in history_files:
            print(f"  - {hf.name}")
    except Exception as e:
        print(f"Could not list history reports: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()