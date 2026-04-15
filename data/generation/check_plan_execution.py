import re
from lxml import etree as LET
import json
import sys
import argparse

def parse_xml_fragment(fragment: str) -> LET._Element:
    """
    Parse XML
    """
    return LET.fromstring(fragment.encode('utf-8'))


EXEC_STEP_RE = re.compile(r"^\s*Transient Step\s+(\d+)\s*:\s*(.*?)\s*->\s*(.*?)\s*$")

def validate_execution_xml(exec_root: LET._Element) -> dict:
    """
      - Root tag must be <Execution>.
      - Must contain one or more <Step> children only.
      - The first line of each <Step> must match 'Transient Step N: <Left> -> <Right>'
      - Collect step numbers and titles.
    """
    if exec_root.tag != 'Execution':
        raise ValueError("Execution validation failed: root element must be <Execution>.")

    steps = list(exec_root.findall('Step'))
    if not steps:
        raise ValueError("Execution validation failed: <Execution> must contain at least one <Step>.")

    numbers = []
    titles = {}

    for i, step in enumerate(steps, 1):
        for child in step:
            raise ValueError(f"Execution validation failed: <Step> must not contain child elements (found <{child.tag}>).")

        text = (step.text or "").strip()
        if not text:
            raise ValueError(f"Execution validation failed: <Step> #{i} is empty.")

        first_line = text.splitlines()[0].strip()
        m = EXEC_STEP_RE.match(first_line)
        if not m:
            raise ValueError(
                f"Execution validation failed: The first line of <Step> #{i} must match "
                f"'Transient Step N: X -> Y', got: {first_line!r}"
            )
        n = int(m.group(1))
        left = m.group(2).strip()
        right = m.group(3).strip()

        if n in numbers:
            raise ValueError(f"Execution validation failed: duplicate step number N={n}.")
        numbers.append(n)
        titles[n] = f"{left} -> {right}"

    return {
        'n_steps': len(steps),
        'numbers': numbers,
        'titles': titles,
    }

PLAN_OUTLINE_RE = re.compile(
    r"^\s*Transient Step\s+(\d+)\s*:\s*(.*?)\s*->\s*(.*?)\s*;\s*Dependency\s*:\s*\[([0-9,\s]*)\]\s*$"
)

def validate_plan_xml(plan_root: LET._Element) -> dict:
    """
      - Root tag must be <Plan>.
      - Must contain one or more <Outline> children only.
      - Each <Outline> must match 'Transient Step N: <Left> -> <Right>; Dependency: [a,b,c]'
      - Dependencies must be a comma-separated list of integers or '[]'.
      - Enforce basic dependency sanity: each dependency must be < current step number.
    """
    if plan_root.tag != 'Plan':
        raise ValueError("Plan validation failed: root element must be <Plan>.")

    outlines = list(plan_root.findall('Outline'))
    if not outlines:
        raise ValueError("Plan validation failed: <Plan> must contain at least one <Outline>.")

    numbers = []
    deps_map = {}
    titles = {}

    seen = set()
    for i, outline in enumerate(outlines, 1):
        for child in outline:
            raise ValueError(f"Plan validation failed: <Outline> must not contain child elements (found <{child.tag}>).")

        text = (outline.text or "").strip()
        if not text:
            raise ValueError(f"Plan validation failed: <Outline> #{i} is empty.")

        m = PLAN_OUTLINE_RE.match(text)
        if not m:
            raise ValueError(
                "Plan validation failed: <Outline> #{} must match "
                "'Transient Step N: X -> Y; Dependency: [a,b]', got: {!r}".format(i, text)
            )

        n = int(m.group(1))
        left = m.group(2).strip()
        right = m.group(3).strip()
        dep_str = m.group(4).strip()

        if n in seen:
            raise ValueError(f"Plan validation failed: duplicate step number N={n}.")
        seen.add(n)
        numbers.append(n)
        titles[n] = f"{left} -> {right}"

        if dep_str == "":
            deps = []
        else:
            deps = [int(x.strip()) for x in dep_str.split(',') if x.strip()]

        for d in deps:
            if d >= n:
                raise ValueError(
                    f"Plan validation failed: step N={n} has an invalid dependency d={d} "
                    f"(must be < {n})."
                )
        deps_map[n] = deps

    return {
        'n_outlines': len(outlines),
        'numbers': numbers,
        'deps': deps_map,
        'titles': titles,
    }

def validate_transient_prompts(execution_xml_str: str, plan_xml_str: str) -> None:
    """
      - Parse execution_xml_str/plan_xml_str.
      - Validate <Execution> and <Plan>.
      - Check number of steps/outlines, step numbers, titles.
      - Validate that all dependencies refer to existing step numbers.
    """
    try:
        exec_root = parse_xml_fragment(execution_xml_str)
    except LET.XMLSyntaxError as e:
        raise ValueError(f"Execution XML parse error: {e}") from e

    try:
        plan_root = parse_xml_fragment(plan_xml_str)
    except LET.XMLSyntaxError as e:
        raise ValueError(f"Plan XML parse error: {e}") from e

    exec_info = validate_execution_xml(exec_root)
    plan_info = validate_plan_xml(plan_root)

    # Count must match
    if exec_info['n_steps'] != plan_info['n_outlines']:
        raise ValueError(
            f"Consistency check failed: #Execution Steps={exec_info['n_steps']} "
            f"!= #Plan Outlines={plan_info['n_outlines']}."
        )

    # Same step number set (order not required)
    if set(exec_info['numbers']) != set(plan_info['numbers']):
        raise ValueError(
            "Consistency check failed: step numbers differ between Execution and Plan: "
            f"{sorted(exec_info['numbers'])} vs {sorted(plan_info['numbers'])}."
        )

    # (Strict) Title match for each step number
    for n in sorted(exec_info['numbers']):
        t_exec = exec_info['titles'].get(n, "")
        t_plan = plan_info['titles'].get(n, "")
        if t_exec != t_plan:
            raise ValueError(
                f"Consistency check failed: title mismatch at step {n}: "
                f"Execution={t_exec!r} vs Plan={t_plan!r}"
            )

    # Dependencies must reference existing numbers
    all_nums = set(plan_info['numbers'])
    for n, deps in plan_info['deps'].items():
        for d in deps:
            if d not in all_nums:
                raise ValueError(
                    f"Consistency check failed: dependency d={d} for step N={n} "
                    f"is not in the set of step numbers."
                )

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Eligibility of Transient Plan and Transient Execution")
    parser.add_argument("--input_json_path", default="./MedVerse_transient.json")
    parser.add_argument("--output_json_path", default="./MedVerse_xml_checked.json")
    args = parser.parse_args()
    json_path = args.input_json_path
    output_path = args.output_json_path

    with open(json_path, 'r') as f:
        data=json.load(f)
        
    k_list, v_list = [], []
    for idx, item in data.items():
        if "Eligible" not in item:
            item["Eligible"] = 0
        elif item["Eligible"] == 2:
            k_list.append(idx)
            v_list.append(item)
            continue
        execution_str = item['Transient Execution Prompt']
        plan_str = item['Transient Plan Prompt']

        try:
            validate_transient_prompts(execution_str, plan_str)
            print(f"Data {idx} Validation passed: Execution and Plan structures, numbering, and dependencies are consistent.")
            item["Eligible"] = 1
        except ValueError as e:
            print(f"Validation failed: {e} at Data {idx}", file=sys.stderr)
            item["Eligible"] = 0
        
        k_list.append(idx)
        v_list.append(item)

    result={k:v for k,v in zip(k_list,v_list)}
    with open(output_path,'w') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
