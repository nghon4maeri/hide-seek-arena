from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_workspace_files_exist():
    required = [
        "docs/architecture.md",
        "docs/algorithm_summary.md",
        "docs/benchmark_report.md",
        "docs/contribution_log.md",
        "docs/workspace_guide.md",
        "scripts/benchmark_agents.py",
        "scripts/run_smoke_test.py",
        "scripts/export_submission.py",
        "tests/test_workspace_structure.py",
        "tests/test_submission_interface.py",
        "tests/test_runtime_smoke.py",
    ]
    for relative_path in required:
        assert (ROOT / relative_path).exists(), relative_path


def test_required_submission_sandboxes_exist():
    for student_id in ["24127457", "24127192", "24127561", "team_submission"]:
        folder = ROOT / "submissions" / student_id
        assert folder.is_dir(), student_id
        assert (folder / "agent.py").is_file(), student_id
        assert (folder / "README.md").is_file(), student_id
        assert (folder / "NOTES.md").is_file(), student_id
