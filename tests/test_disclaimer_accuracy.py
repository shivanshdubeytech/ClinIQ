import json
import urllib.request
import sys

test_cases = [
    ("Severe chest pain", "I am having severe chest pain radiating to my left arm", True),
    ("Choking emergency", "What to do when someone is choking?", True),
    ("Stroke symptoms", "Signs of stroke and facial numbness", True),
    ("Severe asthma attack", "Acute severe asthma attack and cannot breathe", True),
    ("Chronic asthma (Routine)", "How can I manage chronic asthma?", False),
    ("Diabetes signs (Routine)", "What are the common symptoms of type 2 diabetes?", False),
    ("Blood pressure (Routine)", "What is normal blood pressure?", False),
]

print(f"Running {len(test_cases)} Verification Checks on ClinIQ API...\n")
all_passed = True

for name, query, expected_concerning in test_cases:
    data = json.dumps({"user_id": "verifier", "question": query}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8501/api/query",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    r = json.loads(res.read().decode("utf-8"))
    is_con = r.get("is_concerning")
    conf = r.get("confidence", 0.0)
    has_disc = bool(r.get("disclaimer"))
    has_audio = bool(r.get("audio_url"))
    passed_eval = r.get("passed", False)
    
    status = "PASS" if is_con == expected_concerning and has_disc else "FAIL"
    if status == "FAIL":
        all_passed = False
    
    print(f"[{status}] {name:<26} | Expected: {str(expected_concerning):<5} | Actual: {str(is_con):<5} | Conf: {conf:.2f} | Audio: {str(has_audio):<5}")

print(f"\nResult: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
sys.exit(0 if all_passed else 1)
