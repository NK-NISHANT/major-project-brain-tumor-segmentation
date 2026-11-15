import json

with open(r'C:\Users\ACER\.gemini\antigravity\brain\18c94cb3-f8f2-4510-8fca-848e528b98e7\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if 3760 <= line_num <= 3785:
            obj = json.loads(line)
            idx = obj.get('step_index')
            if 'tool_calls' in obj:
                for tc in obj['tool_calls']:
                    print(f"Line {line_num} (step {idx}) Tool: {tc.get('name')}")
                    args = tc.get('args', {})
                    if 'CommandLine' in args:
                        print("  CMD:\n", args['CommandLine'])
