from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
from coq_dependencies import Dependencies, build_shallow_dependencies
from dot_parsing import CoqGraph
from hashlib import md5


class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RED = '\033[91m'


class Style:
    BRIGHT = '\033[1m'
    RESET_ALL = '\033[0m'


@dataclass
class CoqObject:
    name: str
    signature: str
    # Empty for non-proofs.
    body: str
    # Empty for non-proofs.
    lines_until_here_in_file: str
    # Excludes itself! Ordered list of dependencies.
    # For proofs, may include the body, depending on the generation mode.
    dependencies: list[str]
    notations_needed: set[str]
    in_relative_file: Path

    def is_proof(self) -> bool:
        """Whether this object is a proof."""
        return self.body != ''

    def coqtop_input(self, with_answer: bool = False) -> str:
        """Returns the input to be sent to coqtop."""
        base = f'Set Nested Proofs Allowed.\n{self.lines_until_here_in_file}'
        if with_answer:
            return f"{base}\n{self.body}"
        return base

    def log_name(self) -> str:
        """Returns a name for the log file."""
        path_hash = md5(str(self.in_relative_file).encode(),
                        usedforsecurity=False).hexdigest()[:8]
        if len(self.name) > 200:
            name_hash = md5(self.name.encode(),
                            usedforsecurity=False).hexdigest()
            return f"{self.name[:200]}-{name_hash}-{path_hash}.log"
        return f"{self.name}-{path_hash}.log"

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            'name': self.name,
            'signature': self.signature,
            'body': self.body,
            'lines_until_here_in_file': self.lines_until_here_in_file,
            'dependencies': self.dependencies,
            'notations_needed': list(self.notations_needed),
            'in_relative_file': str(self.in_relative_file),
        }

    @staticmethod
    def from_dict(data: dict) -> CoqObject:
        return CoqObject(
            name=data['name'],
            signature=data['signature'],
            body=data['body'],
            lines_until_here_in_file=data['lines_until_here_in_file'],
            dependencies=data['dependencies'],
            notations_needed=set(data['notations_needed']),
            in_relative_file=Path(data['in_relative_file']),
        )

    def llm_prompt(self, *, no_dependencies: bool, no_lines_before: bool) -> str:
        """Returns a string that can be used as a prompt for the LLM."""
        # Don't pass the body, that is the answer!! Body is still attached
        # to this class for evaluation reasons.
        if no_lines_before:
            line_parts = self.lines_until_here_in_file.splitlines()
            requires_only = '\n'.join(
                line for line in line_parts
                if line.strip().startswith('Require Import') or (
                    line.strip().startswith('Require ') and not line.strip().startswith('Require Export')
                ) or (
                    line.strip().startswith('From ') and 'Export' not in line
                )
            )
            if no_dependencies:
                return f"""
Current file path: {self.in_relative_file}

Your goal is to prove the following theorem or lemma:

<goal_signature>
{self.signature}
</goal_signature>

The imports in the current file are provided below for context:

<imports>
{requires_only}
</imports>

Respond ONLY with one of the following:
- The complete proof body, wrapped strictly between "Proof." and "Qed.", without repeating the signature.
- SEARCH(pattern) where pattern is a Coq expression you would like to search for in the codebase to help you construct the proof.
""".strip()
            else:
                return f"""
Current file path: {self.in_relative_file}

Your goal is to prove the following theorem or lemma:

<goal_signature>
{self.signature}
</goal_signature>

Relevant dependencies from other files:

<dependencies>
{'\n'.join(self.dependencies)}
</dependencies>

Relevant notations needed from other files:

<notations>
{'\n'.join(sorted(self.notations_needed))}
</notations>

The imports in the current file are provided below for context:

<imports>
{requires_only}
</imports>

Respond ONLY with the complete proof body, wrapped strictly between "Proof." and "Qed.", without repeating the signature.
""".strip()

        if no_dependencies:
            return f"""
Current file path: {self.in_relative_file}

Your goal is to prove the following theorem or lemma:

<goal_signature>
{self.signature}
</goal_signature>

The current file content up to (but excluding) this theorem or lemma is provided below for context:

<current_file_content>
{self.lines_until_here_in_file}
</current_file_content>

Respond ONLY with the complete proof body, wrapped strictly between "Proof." and "Qed.", without repeating the signature.
""".strip()
        return f"""
Current file path: {self.in_relative_file}

Your goal is to prove the following theorem or lemma:

<goal_signature>
{self.signature}
</goal_signature>

Relevant dependencies from other files:

<dependencies>
{'\n'.join(self.dependencies)}
</dependencies>

Relevant notations needed from other files:

<notations>
{'\n'.join(sorted(self.notations_needed))}
</notations>

The current file content up to (but excluding) this theorem or lemma is provided below for context:

<current_file_content>
{self.lines_until_here_in_file}
</current_file_content>

Respond ONLY with the complete proof body, wrapped strictly between "Proof." and "Qed.", without repeating the signature.
""".strip()


def _build_coq_objects(
    *,
    file_dependency_graph: CoqGraph,
    relative_files_to_signatures_to_dependencies: dict[str, dict[str, Dependencies]],
    files_to_sig_graphs: dict[str, CoqGraph],
    files_to_sig_names_to_signatures: dict[str, dict[str, str]],
    files_to_sig_names_to_bodies: dict[str, dict[str, str]],
    files_to_notations: dict[str, list[str]],
    concatenate_body: bool = False,
    add_samefile_deps: bool = False,
) -> list[CoqObject]:
    """
    Return a list of CoqObjects with all dependencies resolved and flattened.
    """
    file_order = file_dependency_graph.dependency_ordering()
    file_rank = {f: i for i, f in enumerate(file_order)}
    sig_rank = {f: {s: r for r, s in
                    enumerate(files_to_sig_graphs[f].dependency_ordering())}
                for f in file_order}

    cache: dict[tuple[str, str], list[tuple[str, str]]] = {}

    # THIS VERSION assumes there are no cycles
    # def dfs(file_key: str, sig_name: str) -> list[tuple[str, str]]:
    #     k = (file_key, sig_name)
    #     if k in cache:
    #         return cache[k]

    #     if file_key not in relative_files_to_signatures_to_dependencies:
    #         cache[k] = []
    #         return []

    #     if sig_name not in relative_files_to_signatures_to_dependencies[file_key]:
    #         cache[k] = []
    #         return []

    #     rec = relative_files_to_signatures_to_dependencies[file_key][sig_name]

    #     children: list[tuple[str, str]] = []

    #     for dep in sorted(rec.in_the_file, key=lambda s: sig_rank[file_key].get(s, 10**9)):
    #         children.append((file_key, dep))
    #         children += dfs(file_key, dep)

    #     for other_file in sorted(rec.in_other_files, key=lambda f: file_rank[f]):
    #         for dep in sorted(rec.in_other_files[other_file], key=lambda s: sig_rank[other_file].get(s, 10**9)):
    #             # Add the cross-file dependency itself
    #             children.append((other_file, dep))
    #             # Add ALL its dependencies (both in-file and cross-file)
    #             children += dfs(other_file, dep)

    #     seen: set[tuple[str, str]] = set()
    #     ordered = []
    #     for p in children:
    #         if p not in seen and p != k:
    #             ordered.append(p)
    #             seen.add(p)

    #     cache[k] = ordered
    #     return ordered
    def dfs(file_key: str, sig_name: str, visited: Optional[set[tuple[str, str]]] = None) -> list[tuple[str, str]]:
        if visited is None:
            visited = set()

        k = (file_key, sig_name)

        if k in visited:
            return []

        if k in cache:
            return cache[k]

        if file_key not in relative_files_to_signatures_to_dependencies:
            cache[k] = []
            return []

        if sig_name not in relative_files_to_signatures_to_dependencies[file_key]:
            cache[k] = []
            return []

        visited.add(k)

        rec = relative_files_to_signatures_to_dependencies[file_key][sig_name]

        children: list[tuple[str, str]] = []

        for dep in sorted(rec.in_the_file, key=lambda s: sig_rank[file_key].get(s, 10**9)):
            children.append((file_key, dep))
            children += dfs(file_key, dep, visited)

        for other_file in sorted(rec.in_other_files, key=lambda f: file_rank[f]):
            for dep in sorted(rec.in_other_files[other_file], key=lambda s: sig_rank[other_file].get(s, 10**9)):
                children.append((other_file, dep))
                children += dfs(other_file, dep, visited)

        visited.remove(k)

        seen: set[tuple[str, str]] = set()
        ordered = []
        for p in children:
            if p not in seen and p != k:
                ordered.append(p)
                seen.add(p)

        cache[k] = ordered
        return ordered

    objects: list[CoqObject] = []

    for f in file_order:
        sig_map = relative_files_to_signatures_to_dependencies[f]
        sig_items = sig_map.items()
        for sig_name, _ in sig_items:
            ordered_pairs = dfs(f, sig_name)

            # Right now when prompting the LLM for a proof, we provide it with
            # whatever comes before the proof in the same file, plus the dependencies.
            # However if we compute the dependencies naturally same of them are going
            # to be in the same file. So, we'll be including the in-file dependencies
            # twice (one when providing the current file the LLM is working in,
            # and one when providing the dependencies). If we provide certain dependencies
            # twice it'll be easier for the LLM to figure out the parts of the proof.
            #
            # That's why we filter out the current file from the dependencies:
            if add_samefile_deps:
                filtered_pairs = ordered_pairs
            else:
                filtered_pairs = [(pf, ps) for pf, ps in ordered_pairs if pf != f]

            dep_texts = []
            for pf, ps in filtered_pairs:
                r = _render_sig_or_sig_plus_body(
                    file_key=pf,
                    sig_name=ps,
                    concat_body=concatenate_body,
                    sig_texts=files_to_sig_names_to_signatures,
                    body_texts=files_to_sig_names_to_bodies,
                )

                if r is not None:
                    dep_texts.append(r)

            needed_nots: set[str] = set(files_to_notations.get(f, []))
            for pf, _ in ordered_pairs:
                needed_nots.update(files_to_notations.get(pf, []))

            body_here = files_to_sig_names_to_bodies[f].get(sig_name, "")

            objects.append(
                CoqObject(
                    name=sig_name,
                    signature=files_to_sig_names_to_signatures[f][sig_name],
                    body=body_here,
                    lines_until_here_in_file=relative_files_to_signatures_to_dependencies[
                        f][sig_name].lines_until_here_in_file,
                    dependencies=dep_texts,
                    notations_needed=needed_nots,
                    in_relative_file=Path(f),
                )
            )

    return objects


def _render_sig_or_sig_plus_body(
    file_key: str,
    sig_name: str,
    concat_body: bool,
    sig_texts: dict[str, dict[str, str]],
    body_texts: dict[str, dict[str, str]],
) -> Optional[str]:
    if sig_name.startswith('Build_'):
        sig_name = sig_name.removeprefix('Build_')

    try:
        sig_src = sig_texts[file_key][sig_name]
        if not concat_body or sig_name not in body_texts[file_key]:
            return sig_src
        body_src = body_texts[file_key][sig_name]
        return f"{sig_src}\n{body_src}".rstrip()
    except KeyError:
        # print(f"Warning: {file_key} does not have signature {sig_name}")

        # This is not a problem we have to worry about. Often this is caused
        # by record fields, which are not included direclty in the dependency graph.
        # But, they are actually included by transition. For example:
        #
        #
        # Program Instance Monad__Either {e : Type} : GHC.Base.Monad (Either e) :=
        #   fun _ k__ =>
        #     k__ {| GHC.Base.op_zgzg____ := fun {a : Type} {b : Type} =>
        #              Monad__Either_op_zgzg__ ;
        #            GHC.Base.op_zgzgze____ := fun {a : Type} {b : Type} =>
        #              Monad__Either_op_zgzgze__ ;
        #            GHC.Base.return___ := fun {a : Type} => Monad__Either_return_ |}.
        #
        #
        # depends on GHC.Base.op_zgzgze____, that is not in the dependency graph.
        # But, GHC.Base.Monad IS in the dependency graph. So it depends on
        # Monad, which itself depends on Monad__Dict, which has op_zgzgze____.
        return None


def build_coq_objects(
    *,
    project_path: Path,
    logs_dir: Path,
    add_samefile_deps: bool = False,
):
    file_dependency_graph, relative_files_to_signatures_to_dependencies, files_to_sig_graphs, files_to_sig_names_to_bodies, files_to_sig_names_to_signatures, files_to_notations = build_shallow_dependencies(
        project_path
    )

    coq_objects = _build_coq_objects(
        file_dependency_graph=file_dependency_graph,
        relative_files_to_signatures_to_dependencies=relative_files_to_signatures_to_dependencies,
        files_to_sig_graphs=files_to_sig_graphs,
        files_to_sig_names_to_signatures=files_to_sig_names_to_signatures,
        files_to_sig_names_to_bodies=files_to_sig_names_to_bodies,
        files_to_notations=files_to_notations,
        concatenate_body=True,
        add_samefile_deps=add_samefile_deps,
    )

    logs_dir.mkdir(parents=True, exist_ok=True)

    if add_samefile_deps:
        objects_dir = logs_dir / 'objects-same-file-dependencies'
    else:
        objects_dir = logs_dir / 'objects'
    objects_dir.mkdir(parents=True, exist_ok=True)

    for obj in coq_objects:
        with open(objects_dir / obj.log_name(), 'w+') as log_file:
            coqobject_json = json.dumps(
                obj.to_dict(), indent=4, sort_keys=True
            )
            log_file.write(coqobject_json + '\n')

    return coq_objects


def pretty_print(coq_objects: list[CoqObject]):
    proofs_only = [
        coq_object for coq_object in coq_objects if coq_object.is_proof()]
    total_proof = len(proofs_only)
    print(f"==== COQ OBJECTS: {len(coq_objects)}:{total_proof} ====")

    for obj in proofs_only:
        print(f"{Colors.CYAN}{Style.BRIGHT}Name: {obj.name}{Style.RESET_ALL}")
        print(
            f"{Colors.YELLOW}In relative file: {obj.in_relative_file}{Style.RESET_ALL}\n")

        print(f"{Colors.GREEN}{Style.BRIGHT}Signature:{Style.RESET_ALL}")
        print(f"{Colors.GREEN}{obj.signature}{Style.RESET_ALL}")

        if obj.body:
            print(f"\n{Colors.MAGENTA}{Style.BRIGHT}{'='*18}")
            print(f"Body:{Style.RESET_ALL}")
            print(f"{Colors.MAGENTA}{obj.body}{Style.RESET_ALL}")
            print(f"{Colors.MAGENTA}{Style.BRIGHT}{'='*18}{Style.RESET_ALL}")

        # print("Lines until here in file")
        # print()
        # print("==================")
        # print(obj.lines_until_here_in_file)
        # print("==================")
        # print()

        print(
            f"\n{Colors.BLUE}{Style.BRIGHT}Dependencies: {len(obj.dependencies)}{Style.RESET_ALL}")
        print()
        print(f"{Colors.BLUE}{Style.BRIGHT}{'='*18}{Style.RESET_ALL}")
        for i, dep in enumerate(obj.dependencies, 1):
            print(f"{Colors.BLUE}  {i}. {dep}{Style.RESET_ALL}\n")
        print(f"{Colors.BLUE}{Style.BRIGHT}{'='*18}{Style.RESET_ALL}")
        print()

        # print(f"Notations needed: {len(obj.notations_needed)}")
        # for notation in obj.notations_needed:
        #     print(f"  - {notation}")

        print(f"{Colors.RED}{Style.BRIGHT}Press Enter to continue...{Style.RESET_ALL}")
        input()
        print('\n' * 2)
