"""
Every stage the page can run, as a recipe: the script and what it reads, writes and fetches

The order within a table is section 9 of CLAUDE.md, and a job built from several stages keeps it
A job that includes the snapshot stage first puts a staged snapshot in place, so the two never drift
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jobs import Step

P = "papers_preprocess"
T = "topic_tree_creation"
C = "citation_preprocess"
E = "recipe_tuning"
X = "export"

# rows per second measured on this box, replaced by what a finished run here measured
RATES = {
    "papers.1": 3.49,
    "papers.2": 5.91,
    "tree.0": 4167.0,
    "citations.1": 1762.0,
    "citations.relabel": 1762.0,
}
# resolved citations per paper across the corpus, which is what a delta's labeling queue grows by
CITES_PER_PAPER = 23.6
# fixed costs that do not scale with the queue, in seconds
FIXED = {"tree.1": 480.0}


@dataclass
class Stage:
    """
    One script as the page runs it, and what running it does to the store and the disk
    """

    key: str
    table: str
    label: str
    script: str
    args: list[str] = field(default_factory=list)
    reads: str = ""
    writes: str = ""
    fetches: str = ""
    queue: str = ""
    note: str = ""
    resume_args: list[str] | None = None
    delta: bool = True
    # script, or fetch for the one stage the manager runs in process
    kind: str = "script"
    unit: str = "rows"
    # the disks it reads or fills, by the names the health check gives them
    disks: tuple[str, ...] = ("State",)
    # runs only from its own button, never as part of a table or of every stage
    alone: bool = False

    def command(self) -> str:
        """
        The command line the runner launches, as the page prints it

        Returns:
            The command, relative to the checkout.
        """
        if self.kind == "fetch":
            return "GET <SOURCE_URL>, then unzip to arxiv-new.json and compare it with the store"
        return " ".join(
            ["python", f"scripts/{self.script}", *self.args, "--config_path", "<config>"]
        )


STAGES = [
    Stage(
        key="papers.fetch",
        table="papers",
        label="Fetch the snapshot",
        script="",
        reads="Kaggle's Cornell-University/arxiv dataset, which publishes a version every Saturday",
        writes="arxiv-new.json beside arxiv.json, which stays in use until a commit",
        fetches="archive.zip, about 1.8 GB, which unzips to about 5.6 GB",
        queue="remote",
        note="a stopped download resumes where it stopped, and the check runs once it lands",
        delta=False,
        kind="fetch",
        unit="bytes",
        disks=("State", "Snapshot"),
    ),
    Stage(
        key="papers.0",
        table="papers",
        label="Add new papers",
        script=f"{P}/stage0_fetch_source_json.py",
        args=["--no-fetch"],
        reads="the snapshot, arxiv.json, against every id in papers",
        writes="papers: a row per new id, and stage 0 again for each paper with a new version",
        fetches="nothing, the snapshot is fetched in the stage before",
        queue="snapshot",
        note="a new version also deletes that paper's files and its outgoing citations",
        disks=("State", "Snapshot"),
    ),
    Stage(
        key="papers.1",
        table="papers",
        label="Download PDFs",
        script=f"{P}/stage1_download_pdf.py",
        reads="papers at stage 0",
        writes="papers.stage to 1, and a pdf per paper under the corpus",
        fetches="PDFs from the arXiv bulk mirror, falling back to v1 when the latest is missing",
        queue="papers.0",
        disks=("State", "PDF store"),
    ),
    Stage(
        key="papers.2",
        table="papers",
        label="Download HTML",
        script=f"{P}/stage2_download_html.py",
        reads="papers at stage 1",
        writes="papers.stage to 2, and an html page per paper that ar5iv or arxiv.org rendered",
        fetches="HTML from ar5iv, then arxiv.org, at HTML_RATE a second",
        queue="papers.1",
        note="about 60% land, and a paper with no render stays queued for the next run",
        disks=("State", "HTML store"),
    ),
    Stage(
        key="papers.3",
        table="papers",
        label="Embed abstracts",
        script=f"{P}/stage3_embed_papers.py",
        reads="papers at stage 2",
        writes="papers.embed_full and papers.stage to 3",
        fetches="nothing, the embedding endpoint is local",
        queue="papers.2",
    ),
    Stage(
        key="papers.4",
        table="papers",
        label="Score for reviews",
        script=f"{P}/stage4_label_survey.py",
        reads="papers at stage 3",
        writes="papers.survey_probability and papers.stage to 4",
        fetches="nothing, the classifier runs on the GPU",
        queue="papers.3",
    ),
    Stage(
        key="tree.0",
        table="tree",
        label="Project with UMAP",
        script=f"{T}/stage0_umap.py",
        reads="every paper at stage 3 or later",
        writes="topic_tree, rebuilt whole with one 50-dimensional projection per paper at stage 0",
        fetches="nothing",
        queue="embedded",
        note="refuses to start while any paper sits at stage 2",
    ),
    Stage(
        key="tree.1",
        table="tree",
        label="Rebuild the topic tree",
        script=f"{T}/stage1_create_topic_tree.py",
        reads="topic_tree at stage 0",
        writes="topic_tree levels 0, 1 and 2 for every paper, and stage to 1",
        fetches="nothing",
        queue="tree.0",
        note="reclusters the whole corpus, since a partial fit would renumber the leaves",
    ),
    Stage(
        key="tree.2",
        table="tree",
        label="Name the topics",
        script=f"{T}/stage2_topic_labeling.py",
        reads="topic_tree at stage 1 and a sample of titles per leaf",
        writes="topic_names, all 4,368 nodes renamed",
        fetches="nothing, the language model is local",
        queue="names",
        note="refuses to start while any row sits at stage 0",
    ),
    Stage(
        key="citations.0",
        table="citations",
        label="Extract citations",
        script=f"{C}/stage0_create_citation.py",
        args=["--workers", "16"],
        reads="the html of papers at stage 3 or later that were never parsed",
        writes="citations: one row per citation occurrence, and papers.is_citation_parsable",
        fetches="nothing, it reads the html on disk",
        queue="unparsed",
        disks=("State", "HTML store"),
    ),
    Stage(
        key="citations.1",
        table="citations",
        label="Label citations",
        script=f"{C}/stage1_label_citation.py",
        reads="resolved citations at stage 0",
        writes="citations.intent, intent_score and stage to 1",
        fetches="nothing, the labeler runs on the GPU",
        queue="unlabeled",
    ),
    Stage(
        key="citations.2",
        table="citations",
        label="Export the graph",
        script=f"{C}/stage2_export_graph.py",
        reads="every resolved citation",
        writes="citation_graph_export.npy, which graphiv-api loads at start",
        fetches="nothing",
        queue="export",
        note="graphiv-api reads the new file only after it restarts",
    ),
    Stage(
        key="citations.relabel",
        table="citations",
        label="Relabel every citation",
        script=f"{C}/stage1_label_citation.py",
        args=["--relabel"],
        resume_args=[],
        reads="every resolved citation, after resetting them to stage 0",
        writes="citations.intent, intent_score and stage for every resolved row",
        fetches="nothing",
        queue="resolved",
        note="follows a new labeler checkpoint, and a resumed run keeps the rows already done",
        delta=False,
    ),
    Stage(
        key="eval.0",
        table="eval",
        label="Cache the terms",
        script=f"{E}/stage0_cache_terms.py",
        reads="the train and val splits of graphiv-benchmarks-<cutoff>, and the store per pool",
        writes="one npz per benchmark and split under [recipe_tuning] CACHE_PATH",
        note="a split already cached is skipped, so a new cutoff is what makes it rerun",
        delta=False,
    ),
    Stage(
        key="eval.1",
        table="eval",
        label="Fit the recipes",
        script=f"{E}/stage1_fit_recipes.py",
        reads="the cached terms",
        writes="the weights of the five recipes in retrieval_recipes, and the fit's report",
        note="a deployed RECIPE_PATH is read only, so there Dry run reports and writes nothing",
        delta=False,
    ),
    Stage(
        key="eval.2",
        table="eval",
        label="Evaluate",
        script=f"{E}/stage2_evaluate.py",
        reads="one split of every benchmark, [recipe_tuning] SPLIT",
        writes="benchmarks/data/recipe_tuning/<split>/<benchmark>",
        delta=False,
    ),
    Stage(
        key="export.0",
        table="export",
        label="Dump the store",
        script=f"{X}/stage0_dump_store.py",
        reads="every table in the store, from one snapshot of it",
        writes="graphiv-<date>.dump under EXPORT_PATH, a pg_dump custom archive at zstd:3",
        fetches="nothing",
        queue="store",
        note="the full backup, sentences included, which docs/restore.md restores",
        delta=False,
        disks=("State", "Export"),
    ),
    Stage(
        key="export.1",
        table="export",
        label="Export to Parquet",
        script=f"{X}/stage1_export_parquet.py",
        reads="papers at stage 3 or later, their topics, and the citations between them",
        writes="graphiv-<date>/ under EXPORT_PATH: papers, topics, citations and both classifiers",
        fetches="the classifiers' base config from the hub when it is not cached",
        queue="exportable",
        note="citation sentences stay out, since they are the papers' own text",
        delta=False,
        disks=("State", "Export"),
    ),
    Stage(
        key="export.2",
        table="export",
        label="Export the benchmarks",
        script=f"{X}/stage2_export_benchmarks.py",
        reads="every review dated up to [benchmarks] CUTOFF and its resolved references",
        writes="graphiv-benchmarks-<cutoff>/ under EXPORT_PATH, a Parquet per benchmark and split",
        fetches="one question per case from the router, cached so a rerun asks only for new cases",
        queue="",
        note="Dry run counts the cases and their splits without writing",
        delta=False,
        disks=("State", "Export"),
    ),
    Stage(
        key="export.3",
        table="export",
        label="Publish to Hugging Face",
        script=f"{X}/stage3_publish_hf.py",
        reads="the newest graphiv-<date>/ and graphiv-benchmarks-<cutoff>/, and the cards",
        writes="nothing on this machine",
        fetches="uploads every changed file to the four repos in [export], then tags them",
        queue="",
        note="needs [manager] HF_TOKEN_PATH, and Dry run lists the changes without uploading",
        delta=False,
        # it writes nothing here, so a full disk is no reason to wait
        disks=(),
        alone=True,
    ),
]
BY_KEY = {s.key: s for s in STAGES}
TABLES = {
    "papers": "Papers",
    "tree": "Topic tree",
    "citations": "Citations",
    "eval": "Recipe tuning",
    "export": "Export",
}


def step(stage: Stage, extra: list[str] | None = None) -> Step:
    """
    One stage as a job step

    Args:
        stage: The stage.
        extra: Flags the page added.

    Returns:
        The step.
    """
    return Step(
        label=stage.label,
        script=stage.script,
        args=[*stage.args, *(extra or [])],
        kind=stage.kind,
        resume_args=stage.resume_args,
        gate=stage.key == "papers.0",
        stage=stage.key,
    )


def commit() -> Step:
    """
    Puts the staged snapshot in place before the snapshot stage reads it

    Returns:
        The step.
    """
    return Step(label="Put the staged snapshot in place", kind="commit")


def steps(keys: list[str], staged: bool, extra: list[str] | None = None) -> list[Step]:
    """
    The steps for some stages, in recipe order, with the commit first when a snapshot is staged

    Args:
        keys: The stages asked for.
        staged: Whether a snapshot waits to be put in place.
        extra: Flags for a single stage.

    Returns:
        The steps.
    """
    chosen = [s for s in STAGES if s.key in keys]
    out = [commit()] if staged and any(s.key == "papers.0" for s in chosen) else []
    return out + [step(s, extra if len(chosen) == 1 else None) for s in chosen]


def of_table(table: str) -> list[str]:
    """
    The stages of one table that a delta runs

    Args:
        table: papers, tree, citations, eval or export.

    Returns:
        Their keys in order.
    """
    whole = table in ("eval", "export")
    return [s.key for s in STAGES if s.table == table and (s.delta or whole) and not s.alone]


def everything() -> list[str]:
    """
    Every stage a delta runs, which is every table in order and the export last

    Returns:
        Their keys.
    """
    return [s.key for s in STAGES if s.delta]


def waiting(stage: Stage, queues: dict[str, Any], stats: dict[str, Any], found: int) -> int | None:
    """
    How many rows a stage would work through if it started now

    Args:
        stage: The stage.
        queues: The queue counts.
        stats: The cached dashboard numbers.
        found: New plus reset papers in the staged snapshot, zero when nothing is staged.

    Returns:
        The count, None when it cannot be known.
    """
    papers = queues.get("papers", {})
    tree = queues.get("tree", {})
    kind, _, spot = stage.queue.partition(".")
    if stage.queue == "snapshot":
        return found or None
    if stage.queue == "remote":
        return None
    if kind == "papers":
        return int(papers.get(spot, 0))
    if kind == "tree":
        return int(tree.get(spot, 0))
    if stage.queue == "embedded":
        return queues.get("embedded")
    if stage.queue == "names":
        return queues.get("names")
    if stage.queue in ("unparsed", "unlabeled"):
        return queues.get(stage.queue)
    if stage.queue == "store":
        held = stats.get("s_tables", {}).get("rows") or []
        return sum(int(row[4]) for row in held) or None
    if stage.queue == "exportable":
        totals = dict(stats.get("c_totals", {}).get("rows") or [])
        embedded = queues.get("embedded")
        if embedded is None or "resolved" not in totals:
            return None
        return 2 * int(embedded) + int(totals["resolved"])
    if stage.queue in ("resolved", "export"):
        totals = dict(stats.get("c_totals", {}).get("rows") or [])
        return int(totals["resolved"]) if "resolved" in totals else None
    return None


def forecast(
    queues: dict[str, Any], stats: dict[str, Any], found: int, rates: dict[str, float]
) -> list[dict[str, Any]]:
    """
    What a full run would work through at each stage once the staged snapshot is in place

    Args:
        queues: The queue counts now.
        stats: The cached dashboard numbers.
        found: New plus reset papers the snapshot check reported.
        rates: Rows a second per stage, measured here or from the recipe.

    Returns:
        One entry per delta stage with its row count and an estimate in seconds.
    """
    papers = queues.get("papers", {})
    ahead = {
        "papers.0": found,
        "papers.1": int(papers.get("0", 0)) + found,
        "papers.2": int(papers.get("1", 0)) + found,
        "papers.3": int(papers.get("2", 0)) + found,
        "papers.4": int(papers.get("3", 0)) + found,
        "tree.0": int(queues.get("embedded") or 0) + found,
        "tree.1": int(queues.get("embedded") or 0) + found,
        "tree.2": int(queues.get("names") or 0),
        "citations.0": int(queues.get("unparsed") or 0) + found,
        "citations.1": int(queues.get("unlabeled") or 0) + round(found * CITES_PER_PAPER),
        "citations.2": waiting(BY_KEY["citations.2"], queues, stats, 0) or 0,
    }
    out = []
    for key in everything():
        rows = ahead.get(key, 0)
        rate = rates.get(key)
        seconds = FIXED.get(key) or (rows / rate if rate and rows else None)
        out.append({"key": key, "label": BY_KEY[key].label, "rows": rows, "seconds": seconds})
    return out


def plan(
    keys: list[str],
    staged: bool,
    queues: dict[str, Any],
    stats: dict[str, Any],
    found: int,
    remote: dict[str, Any] | None = None,
) -> str:
    """
    What running some stages would do, printed rather than run, which is the page's dry run

    Args:
        keys: The stages.
        staged: Whether a snapshot waits to be put in place.
        queues: The queue counts now.
        stats: The cached dashboard numbers.
        found: New plus reset papers in the staged snapshot.
        remote: The version Kaggle serves, for the fetch stage.

    Returns:
        The plan as console text.
    """
    chosen = [s for s in STAGES if s.key in keys]
    lines = [
        f"dry run: {len(chosen)} stage{'s' if len(chosen) != 1 else ''}, nothing is started",
        "",
    ]
    if staged and any(s.key == "papers.0" for s in chosen):
        lines += [
            "== put the staged snapshot in place",
            "   replaces arxiv.json with arxiv-new.json",
            "",
        ]
    for spot, stage in enumerate(chosen, 1):
        count = waiting(stage, queues, stats, found)
        lines.append(f"== {spot}. {stage.label}")
        lines.append(f"   $ {stage.command()}")
        if stage.kind == "fetch":
            held = remote or {}
            lines.append(
                f"   serves    version {held.get('version')}, published {held.get('published')}"
                if held
                else "   serves    Kaggle's metadata api did not answer"
            )
        else:
            lines.append(
                f"   waiting   {count:,}" if count is not None else "   waiting   not counted yet"
            )
        lines.append(f"   reads     {stage.reads}")
        lines.append(f"   writes    {stage.writes}")
        if stage.fetches:
            lines.append(f"   fetches   {stage.fetches}")
        if stage.note:
            lines.append(f"   note      {stage.note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def recipe() -> list[dict[str, Any]]:
    """
    Every stage as the page reads it

    Returns:
        One dict per stage with its command.
    """
    return [{**asdict(s), "command": s.command()} for s in STAGES]
