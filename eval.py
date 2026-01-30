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


def estimate_eval_input_tokens(coq_objects: list[CoqObject], no_dependencies: bool, no_lines_before: bool) -> int:
    """Estimate the total input token cost for the experiment."""
    return sum(
        count_tokens(SYSTEM_PROMPT + "\n" + coq_object.llm_prompt(
            no_dependencies=no_dependencies, no_lines_before=no_lines_before
            )
        )
        for coq_object in coq_objects if coq_object.is_proof()
    )


def estimate_eval_output_tokens(coq_objects: list[CoqObject], max_tokens: int, bound: Literal['upper', 'lower']) -> int:
    """
    Estimate the total output token cost for the experiment based on the ORIGINAL ANSWERS.

    Note that for reasoning models the estimation won't be as correct. This only estimates
    the final output, and based on the asnwers.
    """
    if bound == 'lower':
        return sum(
            count_tokens(coq_object.body)
            for coq_object in coq_objects if coq_object.is_proof()
        )
    else:
        return sum(
            max_tokens
            for coq_object in coq_objects if coq_object.is_proof()
        )


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
    max_attempts: int = 5  # New parameter for refinement depth
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
        # On subsequent attempts, we guide the LLM to fix the error
        current_prompt = conversation_history
        if attempt > 0:
            current_prompt += f"\n\n[PREVIOUS ATTEMPT]\n{llm_response}\n\n[SYSTEM FEEDBACK]\nThe above attempt failed with the following error:\n{last_error_info}\n\nPlease analyze the error and provide a corrected, complete proof."

        llm_response = call_llm(
            current_prompt,
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

        # Validate the current response
        success, err_type, err_msg = proof_passes(coq_object, llm_response, sertop_args, project_dir)

        if success:
            return success, err_type, err_msg

        # If it failed, update the error info for the next loop iteration
        last_error_info = f"Error Type: {err_type}\nMessage: {err_msg}"

        # Optional: update history so the LLM sees what it tried before
        #conversation_history += f"\n\n[PREVIOUS ATTEMPT]\n{llm_response}"

    # If we exhaust attempts
    return False, 'refinement_failed', f"Exhausted {max_attempts} attempts. Last error: {last_error_info}"


def proof_passes(
    coq_object: CoqObject,
    llm_response: str,
    sertop_args: list[str],
    project_dir: Path
) -> tuple[bool, str, str]:
    """
    Whether the proof passes without any goals remaining, as well
    as the type and error message if any, respectively.
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

        coq_code = coq_object.coqtop_input(
            with_answer=False
        )  # + coq_object.body
        # print()
        # print(coq_code)
        # print()

        add_cmd = sexpdata.dumps(['Add', [], coq_code])
        # print()
        # print(add_cmd)
        # print()

        proc.stdin.write(add_cmd + '\n')
        proc.stdin.flush()

        add_responses = parse_sertop_responses(proc)

        last_sid = -1
        # [Symbol('Answer'), 0, [Symbol('Added'), 34, '[LOC]', Symbol('NewTip')]]
        for response in add_responses:
            if isinstance(response, list) and len(response) > 2:
                nested = response[2]
                if isinstance(nested, list) and len(nested) > 1 and nested[0] == sexpdata.Symbol('Added'):
                    sid = nested[1]
                    if isinstance(sid, int) and sid > last_sid:
                        last_sid = sid
                if isinstance(nested, list) and len(nested) > 0 and nested[0] == sexpdata.Symbol('CoqExn'):
                    # Then we couldn't add the original items? This is a fatal error.
                    # (Never happened in our evaluations)
                    print('Error: Couldn\'t add the original items:', nested)
                    return False, '', ''

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
        add_cmd = sexpdata.dumps(['Add', [], llm_response])
        # print()
        # print(add_cmd)
        # print()
        proc.stdin.write(add_cmd + '\n')
        proc.stdin.flush()

        add_responses = parse_sertop_responses(proc)

        last_sid = -1
        # [Symbol('Answer'), 0, [Symbol('Added'), 34, '[LOC]', Symbol('NewTip')]]
        for response in add_responses:
            if isinstance(response, list) and len(response) > 2:
                nested = response[2]
                if isinstance(nested, list) and len(nested) > 1 and nested[0] == sexpdata.Symbol('Added'):
                    sid = nested[1]
                    if isinstance(sid, int) and sid > last_sid:
                        last_sid = sid
                if isinstance(nested, list) and len(nested) > 0 and nested[0] == sexpdata.Symbol('CoqExn'):
                    # Then we couldn't add the LLM response in the first place (usually a syntax error).
                    # print(nested)
                    for item in nested[1]:
                        if isinstance(item, list) and len(item) > 1 and item[0] == sexpdata.Symbol('str'):
                            return False, 'add_err', item[1].replace('\n', ' ').strip()
                    return False, 'unknown', 'unknown'

        exec_cmd = sexpdata.dumps(['Exec', last_sid])
        # print()
        # print(exec_cmd)
        # print()

        proc.stdin.write(exec_cmd + '\n')
        proc.stdin.flush()
        # exec_responses = parse_sertop_responses(proc)
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(parse_sertop_responses, proc)
            try:
                exec_responses = future.result(timeout=30)
            except futures.TimeoutError:
                # Sometimes the LLM generates proofs that causes Sertop to hang.
                # In that case the proof is simply incorrect.
                print("\nTimeout: sertop did not respond in 30 seconds.")
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                return False, 'timeout', 'Proof timed out'

        # last_response = exec_responses[-1] if exec_responses else None
        # Successful response: exec response: [Symbol('Feedback'), [[Symbol('doc_id'), 0], [Symbol('span_id'), 34], [Symbol('route'), 0], [Symbol('contents'), Symbol('Processed')]]]
        # Failed response:

        # ALSO MODIFY HERE to implement refinement
        for response in exec_responses:
            if isinstance(response, list) and len(response) > 0:
                answer_or_feedback = response[0]

                if answer_or_feedback == sexpdata.Symbol('Feedback'):
                    feedback = response[1]
                    if not feedback_is_ok(feedback):
                        error_message = feedback_message(feedback)
                        # print('Feedback err:', error_message)
                        # Get the erorr message via ^^^ to implement feedback
                        return False, 'feedback', error_message
                elif answer_or_feedback == sexpdata.Symbol('Answer'):
                    answer = response
                    if not answer_is_ok(answer):
                        error_message = answer_message(answer)
                        # print('Answer err:', error_message)
                        # Get the erorr message via ^^^ to implement feedback
                        return False, 'answer', error_message
                else:
                    # This never happened in our evaluations.
                    print("Unknown response type:", answer_or_feedback)
                    print("Do not trust this evaluation! Returning False.")
                    return False, 'unknown_response_type', ''

        result = not admitted(llm_response)
        if not result:
            return False, 'admitted', 'Proof admitted.'
        return True, '', ''

    finally:
        if proc.stdin is not None:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)


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
