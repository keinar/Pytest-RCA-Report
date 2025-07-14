import json
import os
import sys
import re
import threading
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import base64
import io
import numpy as np
import matplotlib.pyplot as plt

# Load environment
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
    sys.exit(1)

# Paths
XML_PATH = './pytest-results.xml'
LOG_PATH = './automation.log'
OUT_JSON = './data/results.json'
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

# Write JSON
Path(OUT_JSON).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"Wrote summary + times + slowest tests JSON to {OUT_JSON}")