from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_blind_workspace_files_exist():
    required = [
        "scripts/benchmark_agents.py",
        "scripts/run_smoke_test.py",
        "scripts/export_submission.py",
        "tests/test_workspace_structure.py",
        "tests/test_submission_interface.py",
        "tests/test_runtime_smoke.py",
    ]
    for relative_path in required:
        assert (ROOT / relative_path).exists(), f"Missing: {relative_path}"


def test_required_blind_submissions_exist():
    for student_id in ["example_student", "team_submission", "24127561", "24127192", "24127457"]:
        folder = ROOT / "submissions" / student_id
        assert folder.is_dir(), f"Missing folder: {student_id}"
        assert (folder / "agent.py").is_file(), f"Missing agent.py: {student_id}"
        assert (folder / "README.md").is_file(), f"Missing README.md: {student_id}"
        assert (folder / "NOTES.md").is_file(), f"Missing NOTES.md: {student_id}"


def test_template_exists():
    assert (ROOT / "submissions" / "TEMPLATE_agent.py").is_file()


def test_broken_agent_exists():
    assert (ROOT / "submissions" / "broken_agent" / "agent.py").is_file()
