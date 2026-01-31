import re
from concurrent import futures
import csv
from pathlib import Path
import subprocess
from typing import Any, Literal
from coq_modules import parse_coq_project_file
from coqobject import CoqObject
from llm import call_llm, count_tokens, SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sexpdata

from models import LLM
from serapi import parse_sertop_responses


MAX_ATTEMPTS = 5


def estimate_eval_input_tokens(
    coq_objects: list[CoqObject], 
    no_dependencies: bool, 
    no_lines_before: bool,
    max_tokens: int = 2048,
    max_attempts: int = MAX_ATTEMPTS
) -> int:
    """
    Estimate total input token cost, accounting for the growing context 
    across multiple refinement attempts.
    """
    total_tokens = 0
    # Heuristic for the size of a typical Coq error message + goal state
    ESTIMATED_FEEDBACK_TOKENS = 350 
    
    for coq_object in coq_objects:
        if not coq_object.is_proof():
            continue
            
        # The base prompt used in Attempt 1
        base_prompt = SYSTEM_PROMPT + "\n" + coq_object.llm_prompt(
            no_dependencies=no_dependencies, no_lines_before=no_lines_before
        )
        base_tokens = count_tokens(base_prompt)
        
        # Each new attempt 'i' includes all previous outputs and feedback.
        # Total Input = Base + (Base + Out1 + Feed1) + (Base + Out1 + Feed1 + Out2 + Feed2) ...
        # Simplified: Total = (max_attempts * base_tokens) + (sum of growing previous context)
        # Using arithmetic progression: (N*(N-1)/2) * (avg_output_size + feedback_size)
        growth_per_step = max_tokens + ESTIMATED_FEEDBACK_TOKENS
        history_penalty = (max_attempts * (max_attempts - 1) // 2) * growth_per_step
        
        total_tokens += (max_attempts * base_tokens) + history_penalty
        
    return int(total_tokens)


def estimate_eval_output_tokens(
    coq_objects: list[CoqObject], 
    max_tokens: int,
    bound: Literal['upper', 'lower'],
    max_attempts: int = MAX_ATTEMPTS
) -> int:
    """
    Estimate total output token cost. Upper bound now scales with attempts.
    """
    proof_objects = [obj for obj in coq_objects if obj.is_proof()]
    
    if bound == 'lower':
        # Assumes every proof passes on the very first attempt
        return sum(count_tokens(obj.body) for obj in proof_objects)
    else:
        # Assumes every proof exhausts the maximum attempts allowed
        return len(proof_objects) * max_tokens * max_attempts


def eval_coq_objects(
    coq_objects: list[CoqObject],
    coq_project_file_path: Path,
    logs_dir: Path,
    *,
    model: LLM,
    no_dependencies: bool,
    no_lines_before: bool,
    max_tokens: int,
    temperature: float,
    thread_count: int,
    do_prints: bool = True
) -> list[bool]:
    """
    Evaluates Coq objects in parallel.
    Only evaluates proof objects, ignores the rest.
    """
    results_dir = model_log_dir(
        logs_dir,
        no_dependencies,
        no_lines_before,
        model,
        max_tokens,
        temperature
    )
    csv_path = results_dir / 'results.csv'

    if csv_path.exists():
        results = csv_path.read_text().strip().split('\n')[1:]
        results = [line.split(',') for line in results]
        results = [line[2] == 'True' for line in results]
        return results

    coq_objects = [obj for obj in coq_objects if obj.is_proof()]

    results = [(False, '', '')] * len(coq_objects)
    # writer = csv.writer(csv_file)
    # writer.writerow(['name', 'file', 'result', 'error_type', 'error'])

    # for coq_object in coq_objects:
    #     if coq_object.name == 'iterates_In':
    #         return [eval_coq_object(
    #             coq_object,
    #             coq_project_file_path,
    #             logs_dir,
    #             model=model,
    #             max_tokens=max_tokens,
    #             temperature=temperature
    #         )]

    def evaluate_single(index_and_obj: tuple[int, CoqObject]) -> tuple[int, tuple[bool, str, str]]:
        index, coq_obj = index_and_obj
        try:
            result = eval_coq_object(
                coq_obj,
                coq_project_file_path,
                logs_dir,
                no_dependencies=no_dependencies,
                no_lines_before=no_lines_before,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return index, result
        except Exception as e:
            if do_prints:
                print(f"Error evaluating {coq_obj.name}: {e}")
            return index, (False, 'eval_failed', 'eval_failed')

    indexed_objects = list(enumerate(coq_objects))

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        future_to_obj = {
            executor.submit(evaluate_single, item): item
            for item in indexed_objects
        }

        if do_prints:
            progress_bar = tqdm(
                total=len(coq_objects),
                desc="Evaluating proofs",
                unit="proof"
            )

        for future in as_completed(future_to_obj):
            index, result = future.result()
            _, coq_object = future_to_obj[future]
            results[index] = result

            # csv_content += f'{coq_object.name},{coq_object.in_relative_file},{result[0]},{result[1]},{result[2]}\n'

            if do_prints:
                progress_bar.update(1)
                if result[0]:
                    progress_bar.set_postfix(status="✓", name=coq_object.name)
                else:
                    progress_bar.set_postfix(status="✗", name=coq_object.name)

        if do_prints:
            progress_bar.close()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['name', 'file', 'result', 'error_type', 'error'])
        for idx, coq_object in enumerate(coq_objects):
            res = results[idx]
            writer.writerow([
                coq_object.name,
                coq_object.in_relative_file,
                res[0], res[1], res[2]
            ])

    return [result[0] for result in results]


def eval_coq_object(
    coq_object: CoqObject,
    coq_project_file_path: Path,
    logs_dir: Path,
    *,
    no_dependencies: bool,
    no_lines_before: bool,
    model: LLM,
    max_tokens: int,
    temperature: float,
    max_attempts: int = MAX_ATTEMPTS  # New parameter for refinement depth
) -> tuple[bool, str, str]:
    """
    Iteratively calls the LLM and refines the proof based on Coq feedback.
    """
    if not coq_object.is_proof():
        raise ValueError(f'Not a proof: {coq_object.name}')

    logfile_path = model_log_dir(
        logs_dir,
        no_dependencies,
        no_lines_before,
        model,
        max_tokens,
        temperature
    ) / coq_object.log_name()

    sertop_args = parse_coq_project_file(coq_project_file_path, 'sercomp')
    project_dir = coq_project_file_path.parent

    if logfile_path.exists():
        llm_response = logfile_path.read_text().strip()
        return proof_passes(coq_object, llm_response, sertop_args, project_dir)

    # Initial prompt setup
    initial_content = coq_object.llm_prompt(
        no_dependencies=no_dependencies,
        no_lines_before=no_lines_before
    )

    # We maintain a history of the conversation for refinement
    conversation_history = initial_content
    last_error_info = ""

    for attempt in range(max_attempts):
        llm_response = call_llm(
            conversation_history,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_info=f'{coq_object.name} - Attempt {attempt + 1}'
        )

        # Log each attempt (optional: you might want to version these logs)
        log_llm_answer(
            logs_dir=logs_dir,
            no_dependencies=no_dependencies,
            no_lines_before=no_lines_before,
            coq_object=coq_object,
            llm_response=llm_response,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        success, err_type, feedback = proof_passes(coq_object, llm_response, sertop_args, project_dir)

        if success:
            return success, err_type, feedback

        # Update History based on the type of feedback
        if err_type == 'search_results':
            header = "[SYSTEM: SEARCH RESULTS]"
        else:
            header = f"[SYSTEM: ERROR - {err_type.upper()}]"
            last_error_info = feedback

        conversation_history += f"\n\nYOUR ATTEMPT:\n{llm_response}\n\n{header}\n{feedback}\n"
        conversation_history += "\nPlease adjust your proof accordingly."

    return False, 'refinement_failed', f"Exhausted {max_attempts} attempts. Last error: {last_error_info}"


def proof_passes(
    coq_object: CoqObject,
    llm_response: str,
    sertop_args: list[str],
    project_dir: Path
) -> tuple[bool, str, str]:
    """
    Evaluates the proof tactic-by-tactic. 
    Returns (Success, ErrorType, FeedbackMessage).
    """
    cmd = ['sertop', *sertop_args, '--implicit', '--omit_loc', '--print0']
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_dir,
        bufsize=1,
    )

    # print('Evaluating', coq_object.name)

    try:
        if proc.stdout is None or proc.stdin is None:
            raise RuntimeError('sertop produced no stdout or stdin')

        # 1. Load context (Imports, Definitions, Lemma Signature)
        coq_code = coq_object.coqtop_input(
            with_answer=False
        )  # + coq_object.body
        # print()
        # print(coq_code)
        # print()
        add_cmd = sexpdata.dumps(['Add', [], coq_code])
        proc.stdin.write(add_cmd + '\n')
        proc.stdin.flush()
        
        # Initial Exec to set up the proof state
        responses = parse_sertop_responses(proc)
        last_sid = max([r[2][1] for r in responses if isinstance(r, list) and len(r) > 2 and r[2][0] == sexpdata.Symbol('Added')] + [-1])
        exec_cmd = sexpdata.dumps(['Exec', last_sid])
        # print()
        # print(exec_cmd)
        # print()
        proc.stdin.write(exec_cmd + '\n')
        proc.stdin.flush()

        exec_responses = parse_sertop_responses(proc)

        # You could probably bundle the original lines + llm response
        # into one Add command, but this is more useful for debugging
        # and returning error information, as well as using the timeout.

        # 2. Iterate through tactics
        tactics = split_tactics(llm_response)
        for tactic in tactics:
            # INTERCEPT: Search Integration
            if tactic.strip().startswith("Search"):
                query_cmd = sexpdata.dumps(['Query', [], [sexpdata.Symbol('Search'), [tactic]]])
                proc.stdin.write(query_cmd + '\n')
                proc.stdin.flush()
                search_res = parse_sertop_responses(proc)
                return False, 'search_results', f"Search Results for '{tactic}':\n{str(search_res)}"

            # Standard Tactic Execution
            add_cmd = sexpdata.dumps(['Add', [], tactic])
            proc.stdin.write(add_cmd + '\n')
            proc.stdin.flush()

            add_responses = parse_sertop_responses(proc)

            # Find New SID
            current_sid = -1
            for resp in add_responses:
                if isinstance(resp, list) and len(resp) > 2 and resp[2][0] == sexpdata.Symbol('Added'):
                    current_sid = resp[2][1]
                if isinstance(resp, list) and len(resp) > 2 and resp[2][0] == sexpdata.Symbol('CoqExn'):
                    return False, 'syntax_error', f"Syntax error in tactic: {tactic}"

            # Execute Tactic
            proc.stdin.write(sexpdata.dumps(['Exec', current_sid]) + '\n')
            proc.stdin.flush()
            
            exec_responses = parse_sertop_responses(proc)
            for resp in exec_responses:
                # Check for Feedback Errors (Tactic failures)
                if isinstance(resp, list) and resp[0] == sexpdata.Symbol('Feedback'):
                    if not feedback_is_ok(resp[1]):
                        goal_state = get_current_goals(proc)
                        msg = feedback_message(resp[1])
                        return False, 'tactic_failure', f"Tactic '{tactic}' failed: {msg}\nProof State:\n{goal_state}"

        # 3. Final Check: No goals remaining
        if admitted(llm_response):
            return False, 'admitted', 'Proof contains Admitted or Admit.'
        return True, '', ''

    finally:
        if proc.stdin is not None:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)


def split_tactics(llm_response: str) -> list[str]:
    """Splits Coq code into individual tactics based on the period terminator."""
    # Matches a dot followed by whitespace or end of string, avoiding dots in comments
    # This is a heuristic; complex notations might require a more robust parser.
    tactics = re.split(r'\.(?=\s|$)', llm_response)
    return [t.strip() + "." for t in tactics if t.strip()]

def get_current_goals(proc: subprocess.Popen) -> str:
    """Queries sertop for the current proof state (hypotheses and goals)."""
    try:
        query_cmd = sexpdata.dumps(['Query', [], [sexpdata.Symbol('Goals'), []]])
        proc.stdin.write(query_cmd + '\n')
        proc.stdin.flush()
        
        responses = parse_sertop_responses(proc)
        # Simplified extraction: in practice, you'd parse the S-exp for 'fg_goals'
        # For this implementation, we'll return a string representation of the response
        return str(responses).replace('\\n', '\n')
    except Exception as e:
        return f"Could not retrieve goals: {e}"


def admitted(llm_response: str) -> bool:
    """Return true if the LLM has used any 'tricks' to avoid proving the goal."""
    parts = llm_response.split()
    if any(k in parts for k in ('Admitted.', 'Admitted', 'Admit', 'Obligation.', 'Obligation')):
        return True

    return False


def feedback_message(feedback: Any) -> str:
    if isinstance(feedback, list):
        for item in feedback:
            if (isinstance(item, list)
                and item
                    and item[0] == sexpdata.Symbol('contents')):
                contents = item[1]
                for subitem in contents:
                    if (isinstance(subitem, list)
                        and subitem
                            and subitem[0] == sexpdata.Symbol('str')):
                        return subitem[1].replace('\n', ' ').strip()
    return 'Couldn\'t recover: ' + str(feedback).replace('\n', ' ').strip()


def answer_message(answer: Any) -> str:
    if isinstance(answer, list) and len(answer) > 2:
        payload = answer[2]
        if isinstance(payload, list) and payload and payload[0] == sexpdata.Symbol('CoqExn'):
            details = payload[1] if len(payload) > 1 else []
            for item in details:
                if isinstance(item, list) and item and item[0] == sexpdata.Symbol('str'):
                    return item[1].replace('\n', ' ').strip()
            for item in details:
                if isinstance(item, list) and item and item[0] == sexpdata.Symbol('exn'):
                    exn_info = item[1]
                    if isinstance(exn_info, list) and len(exn_info) > 1:
                        inner = exn_info[1]
                        if isinstance(inner, str):
                            return inner.replace('\n', ' ').strip()
                        if isinstance(inner, list) and len(inner) > 1 and isinstance(inner[1], str):
                            return inner[1].replace('\n', ' ').strip()
    return 'Couldn\'t recover: ' + str(answer).replace('\n', ' ').strip()


def feedback_is_ok(feedback: Any) -> bool:
    if not feedback or not isinstance(feedback, list):
        return False

    feedback_contents = None
    for item in feedback:
        if isinstance(item, list) and len(item) > 0:
            name = item[0]
            if name == sexpdata.Symbol('contents'):
                feedback_contents = item[1]
                break

    if feedback_contents is None:
        return False

    if feedback_contents == sexpdata.Symbol('Processed') or feedback_contents == sexpdata.Symbol('AddedAxiom'):
        return True
    elif isinstance(feedback_contents, list) and feedback_contents[0] == sexpdata.Symbol('ProcessingIn'):
        return True
    elif isinstance(feedback_contents, list) and feedback_contents[0] == sexpdata.Symbol('Message'):
        # Modify here to get the error message
        message_level = feedback_contents[1]
        message_severity = message_level[1]
        if message_severity == sexpdata.Symbol('Error'):
            return False
        elif message_severity == sexpdata.Symbol('Warning'):
            return True
        print('For severity:', message_severity)
        print('contents:', feedback_contents)
        print('Returning true')
        return True
    else:
        print("Unknown feedback contents:", feedback_contents)
        print("Returning false")
        return False


def answer_is_ok(answer: Any) -> bool:
    if isinstance(answer, list) and len(answer) > 2:
        if answer[2] == sexpdata.Symbol('Ack'):
            return True
        elif isinstance(answer[2], list):
            if answer[2][0] == sexpdata.Symbol('CoqExn'):
                return False
            else:
                print("1-Unknown answer format:", answer[2])
                print("Returning false", answer[2])
                return False
        else:
            print("Not a list in answer[2]:", answer[2])
            print("Returning false")
            return False
    else:
        print("2-Unknown answer format:", answer)
        print("Returning false", answer)
        return False


def model_log_dir(
    logs_dir: Path,
    no_dependencies: bool,
    no_lines_before: bool,
    model: LLM,
    max_tokens: int,
    temperature: float,
) -> Path:
    if no_dependencies and no_lines_before:
        return logs_dir / f"nolines-nodeps-{model}-{temperature}-{max_tokens}"
    elif no_dependencies:
        return logs_dir / f"nodeps-{model}-{temperature}-{max_tokens}"
    elif no_lines_before:
        return logs_dir / f"nolines-{model}-{temperature}-{max_tokens}"
    else:
        return logs_dir / f"{model}-{temperature}-{max_tokens}"


def log_llm_answer(
    *,
    logs_dir: Path,
    no_dependencies: bool,
    no_lines_before: bool,
    coq_object: CoqObject,
    llm_response: str,
    model: LLM,
    max_tokens,
    temperature,
):
    log_dir_for_model = model_log_dir(
        logs_dir, no_dependencies, no_lines_before, model, max_tokens, temperature
    )
    log_dir_for_model.mkdir(parents=True, exist_ok=True)

    with open(log_dir_for_model / coq_object.log_name(), 'w+') as log_file:
        log_file.write(llm_response + '\n')
